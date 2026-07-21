"""Goal-contract and executive-digest helpers extracted from AutonomousAgentExecutor."""

from __future__ import annotations

from typing import Any, Dict, List


class AgentGoalContractService:
    """Evaluate deterministic completion contracts and build operator digests."""

    def evaluate_goal_contract(
        self,
        executor: Any,
        job: Any,
        state: Dict[str, Any],
        *,
        include_result_keys: bool = True,
    ) -> Dict[str, Any]:
        """Evaluate whether deterministic completion requirements are currently satisfied."""
        contract = executor._get_goal_contract_config(job)
        if not bool(contract.get("enabled", False)):
            return {
                "enabled": False,
                "satisfied": True,
                "missing": [],
                "contract": contract,
                "metrics": {
                    "progress": int(state.get("goal_progress", 0) or 0),
                    "findings_count": len(state.get("findings", []) if isinstance(state.get("findings"), list) else []),
                    "artifacts_count": len(state.get("artifacts", []) if isinstance(state.get("artifacts"), list) else []),
                },
            }

        findings = state.get("findings") if isinstance(state.get("findings"), list) else []
        artifacts = state.get("artifacts") if isinstance(state.get("artifacts"), list) else []
        progress = int(state.get("goal_progress", 0) or 0)

        finding_types: Dict[str, int] = {}
        for finding in findings:
            if not isinstance(finding, dict):
                continue
            ftype = str(finding.get("type") or "").strip()
            if ftype:
                finding_types[ftype] = int(finding_types.get(ftype, 0) or 0) + 1

        artifact_types: Dict[str, int] = {}
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                continue
            atype = str(artifact.get("type") or "").strip()
            if atype:
                artifact_types[atype] = int(artifact_types.get(atype, 0) or 0) + 1

        missing: List[str] = []
        min_progress = int(contract.get("min_progress", 100) or 100)
        min_findings = int(contract.get("min_findings", 0) or 0)
        min_artifacts = int(contract.get("min_artifacts", 0) or 0)

        if progress < min_progress:
            missing.append(f"progress>={min_progress}")
        if len(findings) < min_findings:
            missing.append(f"findings>={min_findings}")
        if len(artifacts) < min_artifacts:
            missing.append(f"artifacts>={min_artifacts}")

        required_finding_types = contract.get("required_finding_types")
        if isinstance(required_finding_types, list):
            for ftype in [str(x).strip() for x in required_finding_types if str(x).strip()]:
                if int(finding_types.get(ftype, 0) or 0) <= 0:
                    missing.append(f"finding_type:{ftype}")

        required_artifact_types = contract.get("required_artifact_types")
        if isinstance(required_artifact_types, list):
            for atype in [str(x).strip() for x in required_artifact_types if str(x).strip()]:
                if int(artifact_types.get(atype, 0) or 0) <= 0:
                    missing.append(f"artifact_type:{atype}")

        required_result_keys = contract.get("required_result_keys")
        if include_result_keys and isinstance(required_result_keys, list):
            results = job.results if isinstance(job.results, dict) else {}
            for key in [str(x).strip() for x in required_result_keys if str(x).strip()]:
                if key not in results:
                    missing.append(f"result_key:{key}")

        return {
            "enabled": True,
            "satisfied": len(missing) == 0,
            "missing": missing[:20],
            "contract": contract,
            "metrics": {
                "progress": progress,
                "findings_count": len(findings),
                "artifacts_count": len(artifacts),
                "finding_types": finding_types,
                "artifact_types": artifact_types,
            },
        }

    def build_executive_digest(self, executor: Any, job: Any, state: Dict[str, Any]) -> Dict[str, Any]:
        """Build a compact deterministic run digest for operators."""
        findings = state.get("findings") if isinstance(state.get("findings"), list) else []
        artifacts = state.get("artifacts") if isinstance(state.get("artifacts"), list) else []
        actions = state.get("actions_taken") if isinstance(state.get("actions_taken"), list) else []
        critic_notes = state.get("critic_notes") if isinstance(state.get("critic_notes"), list) else []

        key_findings: List[str] = []
        for finding in findings:
            if not isinstance(finding, dict):
                continue
            title = str(finding.get("title") or finding.get("summary") or finding.get("id") or "").strip()
            if title and title not in key_findings:
                key_findings.append(title[:220])
            if len(key_findings) >= 5:
                break

        failed_actions = 0
        for row in actions:
            if not isinstance(row, dict):
                continue
            result = row.get("result") if isinstance(row.get("result"), dict) else {}
            if not bool(result.get("success")):
                failed_actions += 1

        risks: List[str] = []
        if failed_actions > 0:
            risks.append(f"{failed_actions} tool actions failed during execution")
        for note in critic_notes[-4:]:
            if not isinstance(note, dict):
                continue
            sev = str(note.get("severity") or "").strip().lower()
            if sev not in {"high", "medium"}:
                continue
            pivot = str(note.get("pivot") or note.get("trajectory_assessment") or "").strip()
            if not pivot:
                continue
            risks.append(f"critic_{sev}: {pivot[:180]}")
            if len(risks) >= 5:
                break

        contract_eval = self.evaluate_goal_contract(executor, job, state)
        if bool(contract_eval.get("enabled")) and not bool(contract_eval.get("satisfied")):
            missing = contract_eval.get("missing") if isinstance(contract_eval.get("missing"), list) else []
            if missing:
                risks.append(f"goal contract unmet: {', '.join([str(x)[:60] for x in missing[:3]])}")

        next_actions: List[str] = []
        causal_plan = state.get("causal_experiment_plan") if isinstance(state.get("causal_experiment_plan"), dict) else {}
        causal_experiments = causal_plan.get("experiments") if isinstance(causal_plan.get("experiments"), list) else []
        causal_priority = causal_plan.get("priority_order") if isinstance(causal_plan.get("priority_order"), list) else []
        exp_index = {
            str(exp.get("id") or "").strip(): exp
            for exp in causal_experiments
            if isinstance(exp, dict) and str(exp.get("id") or "").strip()
        }
        ordered = [str(x).strip() for x in causal_priority if str(x).strip() in set(exp_index.keys())]
        if not ordered:
            ordered = list(exp_index.keys())
        for eid in ordered[:2]:
            exp = exp_index.get(eid) if isinstance(exp_index.get(eid), dict) else {}
            if not exp:
                continue
            name = str(exp.get("name") or "").strip()
            if name:
                next_actions.append(f"Execute causal experiment {eid}: {name[:160]}")
            expected = exp.get("expected_evidence") if isinstance(exp.get("expected_evidence"), dict) else {}
            supports = expected.get("supports") if isinstance(expected.get("supports"), list) else []
            if supports:
                next_actions.append(f"Evidence to confirm: {str(supports[0])[:160]}")
            if len(next_actions) >= 4:
                break

        results = job.results if isinstance(job.results, dict) else {}
        research_bundle = results.get("research_bundle") if isinstance(results.get("research_bundle"), dict) else {}
        rb_steps = research_bundle.get("next_steps") if isinstance(research_bundle.get("next_steps"), list) else []
        for step in rb_steps:
            txt = str(step).strip()
            if txt and txt not in next_actions:
                next_actions.append(txt[:200])
            if len(next_actions) >= 4:
                break
        if not next_actions and bool(contract_eval.get("enabled")) and not bool(contract_eval.get("satisfied")):
            missing = contract_eval.get("missing") if isinstance(contract_eval.get("missing"), list) else []
            for req in missing[:3]:
                next_actions.append(f"Satisfy contract requirement: {str(req)[:120]}")
        if not next_actions:
            next_actions = [
                "Review top findings for consistency and evidence quality.",
                "Plan one focused follow-up iteration on highest-impact gap.",
            ]

        summary = str(results.get("summary") or "").strip()
        if not summary:
            summary = (
                f"Completed {int(job.iteration or 0)} iterations with "
                f"{len(findings)} findings and {len(artifacts)} artifacts."
            )

        return {
            "goal": str(job.goal or "").strip()[:600],
            "status": str(job.status or ""),
            "outcome": summary[:400],
            "metrics": {
                "goal_progress": int(state.get("goal_progress", 0) or 0),
                "iterations": int(job.iteration or 0),
                "actions_count": len(actions),
                "findings_count": len(findings),
                "artifacts_count": len(artifacts),
                "failed_actions": failed_actions,
            },
            "key_findings": key_findings,
            "risks": risks[:5],
            "next_actions": next_actions[:5],
            "goal_contract": {
                "enabled": bool(contract_eval.get("enabled")),
                "satisfied": bool(contract_eval.get("satisfied")),
                "missing": contract_eval.get("missing") if isinstance(contract_eval.get("missing"), list) else [],
            },
        }
