from unittest.mock import patch

import vllm_ascend.models as ascend_models


def test_register_pangu_embedded_model() -> None:
    with patch.object(ascend_models.ModelRegistry, "register_model") as register_model:
        ascend_models.register_model()

    register_model.assert_any_call(
        "PanguEmbeddedForCausalLM",
        "vllm_ascend.models.open_pangu:PanguEmbeddedForCausalLM",
    )