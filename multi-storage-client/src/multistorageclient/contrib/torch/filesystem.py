# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language overning permissions and
# limitations under the License.

import io
import logging
import os
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from typing import IO, Union, cast

import torch
from torch import Tensor
from torch.distributed._shard._utils import narrow_tensor_by_index
from torch.distributed.checkpoint.filesystem import FileSystemBase, FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint.planner import (
    LoadItemType,
    LoadPlan,
    LoadPlanner,
    ReadItem,
)
from torch.futures import Future

from ...pathlib import MultiStoragePath
from ...types import Range

logger = logging.getLogger(__name__)


class MultiStorageFileSystem(FileSystemBase):
    """
    A filesystem implementation that uses the MultiStoragePath class to handle paths.
    """

    @contextmanager
    def create_stream(self, path: Union[str, os.PathLike], mode: str) -> Generator[io.IOBase, None, None]:
        # always download the file here (used for checkpointing)
        with MultiStoragePath(path).open(mode=mode, prefetch_file=True) as fp:
            yield fp

    def concat_path(self, path: Union[str, os.PathLike], suffix: str) -> Union[str, os.PathLike]:
        return MultiStoragePath(path) / suffix

    def rename(self, path: Union[str, os.PathLike], new_path: Union[str, os.PathLike]) -> None:
        MultiStoragePath(path).rename(new_path)

    def init_path(self, path: Union[str, os.PathLike]) -> Union[str, os.PathLike]:
        return MultiStoragePath(path)

    def mkdir(self, path: Union[str, os.PathLike]) -> None:
        MultiStoragePath(path).mkdir(parents=True, exist_ok=True)

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        try:
            MultiStoragePath(checkpoint_id)
        except ValueError:
            return False

        return True

    def exists(self, path: Union[str, os.PathLike]) -> bool:
        return MultiStoragePath(path).exists()

    def rm_file(self, path: Union[str, os.PathLike]) -> None:
        MultiStoragePath(path).unlink()

    def ls(self, path: Union[str, os.PathLike]) -> list[str]:
        return [str(p) for p in MultiStoragePath(path).iterdir()]


def _prefetch_objects(fs: MultiStorageFileSystem, urls: list[MultiStoragePath], thread_count: int) -> None:
    """
    Efficiently pre-downloads files from object storage using parallel threads, storing them in cache when enabled for optimized subsequent access.
    """

    def _prefetch(url: MultiStoragePath) -> None:
        with fs.create_stream(url, "rb") as _:
            pass

    with ThreadPoolExecutor(max_workers=thread_count) as executor:
        futures = [executor.submit(_prefetch, url) for url in urls]
        for future in futures:
            future.result()


class MultiStorageFileSystemReader(FileSystemReader):
    """
    A reader implementation that uses the MultiStorageFileSystem class to handle file system operations.
    """

    def __init__(self, path: Union[str, os.PathLike], thread_count: int = 1) -> None:
        """
        Initialize the MultiStorageFileSystemReader with the MultiStorageFileSystem.

        :param path: The path to the checkpoint.
        :param thread_count: The number of threads to use for prefetching.
        """
        super().__init__(path)
        self.fs = MultiStorageFileSystem()
        self.path = self.fs.init_path(path)
        self.thread_count = thread_count

    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        """
        Read checkpoint data using parallel byte-range GET requests.

        Instead of downloading entire shard files sequentially, this issues
        concurrent range-GET requests for each tensor slice, significantly
        improving throughput for large checkpoints on object storage.

        Falls back to the base class implementation when thread_count <= 1.
        """
        if self.thread_count <= 1:
            return super().read_data(plan, planner)

        per_file: dict[str, list[ReadItem]] = {}
        for read_item in plan.items:
            item_md = self.storage_data[read_item.storage_index]
            per_file.setdefault(item_md.relative_path, []).append(read_item)

        msc_path = cast(MultiStoragePath, self.path)

        def _read_item(relative_path: str, req: ReadItem) -> tuple[ReadItem, bytes]:
            item_md = self.storage_data[req.storage_index]
            file_path = msc_path / relative_path
            data = file_path._storage_client.read(
                str(file_path._internal_path),
                byte_range=Range(offset=item_md.offset, size=item_md.length),
            )
            return (req, data)

        completed = 0

        # Process results as they arrive to avoid buffering all data in memory.
        # At most thread_count fetches are in-flight at once.
        with ThreadPoolExecutor(max_workers=self.thread_count) as executor:
            futures = []
            for relative_path, reqs in per_file.items():
                for req in reqs:
                    futures.append(executor.submit(_read_item, relative_path, req))

            for future in as_completed(futures):
                req, raw_data = future.result()
                self._commit_read_item(req, raw_data, planner)
                completed += 1

        logger.info("Fetched and committed %d tensor slices using %d threads", completed, self.thread_count)

        fut: Future = Future()
        fut.set_result(None)
        return fut

    def _commit_read_item(self, req: ReadItem, raw_data: bytes, planner: LoadPlanner) -> None:
        """Deserialize a single fetched tensor slice and commit it to the planner."""
        item_md = self.storage_data[req.storage_index]
        raw_stream: IO[bytes] = io.BytesIO(raw_data)
        transform_from = self.transforms.transform_load_stream(
            req,
            item_md.transform_descriptors or (),
            raw_stream,
        )

        if req.type == LoadItemType.BYTE_IO:
            read_bytes = io.BytesIO(transform_from.read(-1))
            read_bytes.seek(0)
            planner.load_bytes(req, read_bytes)
        else:
            if transform_from.seekable():
                seekable = transform_from
            else:
                seekable = io.BytesIO(transform_from.read(-1))
                seekable.seek(0)

            tensor = cast(
                Tensor,
                torch.load(seekable, map_location="cpu", weights_only=True),
            )
            tensor = narrow_tensor_by_index(
                tensor, req.storage_offsets, req.lengths
            )
            target_tensor = planner.resolve_tensor(req).detach()
            assert target_tensor.size() == tensor.size(), (
                f"req {req.storage_index} mismatch sizes "
                f"{target_tensor.size()} vs {tensor.size()}"
            )
            target_tensor.copy_(tensor)
            planner.commit_tensor(req, target_tensor)

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        return MultiStorageFileSystem.validate_checkpoint_id(checkpoint_id)


class MultiStorageFileSystemWriter(FileSystemWriter):
    """
    A writer implementation that uses the MultiStorageFileSystem class to handle file system operations.
    """

    def __init__(
        self,
        path: Union[str, os.PathLike],
        single_file_per_rank: bool = True,
        sync_files: bool = True,
        thread_count: int = 1,
        per_thread_copy_ahead: int = 10_000_000,
        cache_staged_state_dict: bool = False,
        overwrite: bool = True,
    ) -> None:
        """
        Initialize the MultiStorageFileSystemWriter with the MultiStorageFileSystem.
        """
        super().__init__(
            path,
            single_file_per_rank,
            sync_files,
            thread_count,
            per_thread_copy_ahead,
            cache_staged_state_dict,
            overwrite=overwrite,
        )
        self.fs = MultiStorageFileSystem()
        self.path = self.fs.init_path(path)

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        return MultiStorageFileSystem.validate_checkpoint_id(checkpoint_id)
