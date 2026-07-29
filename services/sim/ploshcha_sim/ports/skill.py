from abc import ABC, abstractmethod

from ..domain.skill import SkillSpec, scope


class Skill(ABC):
    @property
    @abstractmethod
    def spec(self) -> SkillSpec: ...


class SkillRegistry(ABC):
    @abstractmethod
    def skills(self) -> list[Skill]: ...

    def skill_specs(self) -> list[SkillSpec]:
        return [s.spec for s in self.skills()]

    def capabilities(self) -> list[str]:
        seen: list[str] = []
        for spec in self.skill_specs():
            if spec.capability not in seen:
                seen.append(spec.capability)
        return seen

    def scoped_specs(self, capabilities: tuple[str, ...]) -> list[SkillSpec]:
        return scope(self.skill_specs(), capabilities)

    @abstractmethod
    def scoped(self, capabilities: tuple[str, ...]) -> "SkillRegistry": ...
