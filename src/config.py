from pydantic import BaseModel, model_validator, Field


class AttentionConfig(BaseModel):
    """
    A config pydantic class that checks the following:
        - returns ValueError if d_model is not divisable by n_heads
        - d_heads returns the dimensionality of attention heads
    """

    d_model: int = Field(1024, gt=0)
    n_heads: int = Field(16, gt=0)
    dropout: float = Field(0.1, ge=0.0, lt=1.0)
    max_seq_len: int = Field(1024, gt=0)

    @model_validator(mode="after")
    def heads_divide_model(self) -> "AttentionConfig":
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) should be divisable by the n_heads ({self.n_heads})"
            )
        return self

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads
