# Copyright Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
from typing import List

import torch

from atb_llm.utils.adapter_manager import AdapterIdsType
from atb_llm.models.base.config import BaseConfig
from atb_llm.utils.weights import Weights
from atb_llm.utils.adapter_manager import AdapterManager
from atb_llm.models.base.base_lm import BaseLM


class LoraDecorator:
    """This class contains methods and attributes required to enable Multi-LoRA functionality in a model.

    Attributes:
        lora_adapter: A dict contains adapter id and adapter path.
        active: A flag indicating whether model has lora adapters.
        adapter_manager: A class manages lora weights and adapter metainfo.
        adapter_weights: A list contains currently activated adapter weights.
        _torch_device: Device of weights.
        _torch_dtype: `torch_dtype` from model's config.json.
    """
    def __init__(self, config: BaseConfig, weights: Weights, base_model: BaseLM, **kwargs):
        self.lora_adapter = kwargs.get("lora_adapter")
        self.active = self.lora_adapter is not None
        self.adapter_manager = None
        self.adapter_weights = []

        # Parameters from base class
        self._torch_device = weights.device
        self._torch_dtype = weights.dtype
        self._placeholder = torch.zeros(1, dtype=self._torch_dtype, device="npu")

        if self.active:
            self.adapter_manager = AdapterManager(weights)
            self.adapter_manager.base_model = base_model

    def use_multi_adapters(self) -> bool:
        """
        Determine if multiple adapters are being used.

        Returns:
            bool: True if multiple adapters are being used, False otherwise.
        """
        return self.adapter_manager is not None \
            and self.adapter_manager.previous_adapter_ids is not None \
            and self.adapter_manager.previous_adapter_ids.record_type != AdapterIdsType.SINGLE

    def use_single_adapter(self) -> bool:
        """
        Determine if a single adapter is being used.

        Returns:
            bool: True if a single adapter is being used, False otherwise.
        """
        return self.adapter_manager is not None \
            and self.adapter_manager.previous_adapter_ids is not None \
            and self.adapter_manager.previous_adapter_ids.record_type == AdapterIdsType.SINGLE \
            and self.adapter_manager.previous_adapter_ids.adapter_ids != ["base"]

    def use_no_adapter(self) -> bool:
        """
        Determine if no adapter is being used.

        Returns:
            bool: True if no adapter is being used, False otherwise.
        """
        return self.adapter_manager is not None \
            and self.adapter_manager.previous_adapter_ids is not None \
            and self.adapter_manager.previous_adapter_ids.adapter_ids == ["base"]

    def update_inputs(
            self,
            engine_inputs: List[torch.Tensor],
            adapter_ids: List[str],
            input_lengths: torch.Tensor,
            is_prefill: bool,
            **kwargs
        ) -> None:
        """
        Update the inputs for the engine based on the provided adapter IDs and input lengths.

        `engine_inputs` will be updated with group size and adpater weights as needed.
        group size is created by `_calculate_adapter_group_size`.
        adpater weights will be chosen by `adapter_ids`.

        Args:
            engine_inputs (List[torch.Tensor]): The initial list of engine inputs.
            adapter_ids (List[str]): The list of adapter IDs assigned to each request in the batch.
            input_lengths (torch.Tensor): The lengths of the inputs for each request in the batch.
            is_prefill (bool): The inference phase.
                If `is_prefill` is true, it means the prefill phase; otherwise, it means the decode phase.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        if not self.active:
            return

        effective_adapter_ids = self.adapter_manager.preprocess_adatper_ids(adapter_ids)

        need_update = self.adapter_manager.update_adapter(effective_adapter_ids)
        # no adapter
        if effective_adapter_ids == ["base"]:
            return
        if need_update:
            # 更新adapter weights
            self.adapter_weights = self.adapter_manager.get_adapters(effective_adapter_ids)

        # single adapter or multi adapters
        if is_prefill:
            actual_input_tokens = input_lengths
        else:
            actual_input_tokens = torch.ones_like(input_lengths, device=self._torch_device, dtype=torch.int64)
        engine_inputs.append(self._calculate_adapter_group_size(
            effective_adapter_ids, actual_input_tokens, is_prefill=is_prefill))
        for weight in self.adapter_weights:
            engine_inputs.append(weight)

    def _calculate_adapter_group_size(
            self, adapter_ids: None | List[str | None],
            input_lengths: torch.Tensor, is_prefill: bool = False
    ) -> torch.Tensor:
        """
        Calculate the group size based on the adapter IDs and input lengths.

        group size is used when multiple lora adapters are enabled in one batch.
        It means the cumulative sum of the actual sequence lengths for each 
        matrix passing into the group matmul operation.
        It has different calculation method depends on whether `adapter_ids` is sorted and the inference phase.

        Args:
            adapter_ids (None | List[str | None]): The list of adapter IDs.
            input_lengths (torch.Tensor): The lengths of the inputs.
            is_prefill (bool, optional): Whether to prefill the inputs. Defaults to False.

        Returns:
            torch.Tensor: The calculated adapter group size.
        """
        if len(adapter_ids) == 1:
            return self._placeholder
        elif self.adapter_manager.previous_adapter_ids.record_type == AdapterIdsType.MIXED:
            if is_prefill:
                cum_group_size = torch.cumsum(input_lengths, dim=0, dtype=torch.int64)
            else:
                cum_group_size = torch.arange(1, input_lengths.shape[0] + 1, 
                                              dtype=torch.int64, device=self._torch_device)
        else:
            active_adapters_count = len(self.adapter_manager.adapter_info_registry) - 1  # exclude *sort
            adapter_indexes = []
            for adapter_id in adapter_ids:
                adapter_indexes.append(self.adapter_manager.adapter_info_registry.get(adapter_id).idx)
            labels = torch.tensor(adapter_indexes, device=self._torch_device, dtype=torch.int64)
            unique_labels = torch.arange(0, active_adapters_count, dtype=torch.int64, device=self._torch_device)
            group = torch.zeros_like(unique_labels).scatter_add_(0, labels, input_lengths.to(torch.int64))
            cum_group_size = torch.cumsum(group, dim=0, dtype=torch.int64)
        return cum_group_size
