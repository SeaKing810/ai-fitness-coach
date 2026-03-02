from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .utils import sanitize_text


@dataclass
class UserProfile:
    age: Optional[int]
    experience: str
    goals: str
    constraints: str


class FitnessChatbot:
    def __init__(self, model_name: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()

        if torch.cuda.is_available():
            self.model.to("cuda")

    def _device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    def generate(
        self,
        user_text: str,
        profile: UserProfile,
        short_context: List[Dict[str, Any]],
        extra_context: str = "",
        max_new_tokens: int = 120,
    ) -> str:
        user_text = sanitize_text(user_text)
        goals = sanitize_text(profile.goals)
        exp = sanitize_text(profile.experience)
        constraints = sanitize_text(profile.constraints)
        age = profile.age if profile.age is not None else "unknown"

        context_lines: List[str] = []
        for m in short_context[-8:]:
            role = m.get("role", "")
            content = sanitize_text(m.get("content", ""))
            context_lines.append(f"{role}: {content}")

        system_style = (
            "You are a helpful fitness coach. Keep answers practical, concise, and safe. "
            "Do not claim you are a doctor. If the user reports pain or injury, recommend seeing a professional. "
            "Focus on progressive overload, recovery, and basic technique cues. "
            "Use plain language.\n"
        )

        personalization = f"User profile, age {age}, experience {exp}, goals {goals}, constraints {constraints}.\n"

        stitched = (
            system_style
            + personalization
            + (extra_context.strip() + "\n" if extra_context.strip() else "")
            + "\n".join(context_lines)
            + ("\n" if context_lines else "")
            + f"user: {user_text}\ncoach:"
        )

        inputs = self.tokenizer.encode(stitched, return_tensors="pt").to(self._device())

        with torch.no_grad():
            out = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.92,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        decoded = self.tokenizer.decode(out[0], skip_special_tokens=True)

        if "coach:" in decoded:
            reply = decoded.split("coach:", 1)[-1].strip()
        else:
            reply = decoded.strip()

        reply = reply.split("user:", 1)[0].strip()
        return reply[:900] if reply else "I did not catch that. Tell me your goal and what equipment you have."
