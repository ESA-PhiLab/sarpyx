"""YAML-configured SNAP pipeline runner.

This module is intentionally separate from the existing :mod:`snapflow`
operator wrappers.  It provides an alternate config-driven execution mode that
can dispatch either raw SNAP operators rendered as temporary XML graphs or
existing :class:`sarpyx.snapflow.engine.GPT` methods.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised only in broken envs
    raise RuntimeError(
        "YAML pipeline support requires PyYAML. Install project dependencies with uv."
    ) from exc

from sarpyx.snapflow.engine import GPT


class ConfigPipelineError(ValueError):
    """Raised when a YAML pipeline is invalid or cannot be executed safely."""


@dataclass(frozen=True)
class StepRecord:
    """Execution or dry-run record for one config pipeline step."""

    pipeline: str
    step_id: str
    kind: str
    action: str
    output: str | None
    graph: str | None = None
    source_refs: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineRunResult:
    """Result returned after running or planning a configured pipeline."""

    pipeline: str
    output: str | None
    records: tuple[StepRecord, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "pipeline": self.pipeline,
            "output": self.output,
            "steps": [record.__dict__ for record in self.records],
        }


def load_pipeline_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML pipeline config from disk."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config is None:
        raise ConfigPipelineError(f"Pipeline config is empty: {config_path}")
    if not isinstance(config, dict):
        raise ConfigPipelineError("Pipeline config root must be a mapping")
    return config


def list_config_pipelines(config: Mapping[str, Any]) -> list[str]:
    """Return pipeline names defined by a loaded config."""

    pipelines = config.get("pipelines")
    if not isinstance(pipelines, dict):
        raise ConfigPipelineError("Config must contain a 'pipelines' mapping")
    return sorted(str(name) for name in pipelines)


def validate_pipeline_config(config: Mapping[str, Any]) -> None:
    """Validate config structure and named pipeline composition."""

    version = config.get("version")
    if version != 1:
        raise ConfigPipelineError("Only pipeline config version 1 is supported")

    defaults = config.get("defaults", {})
    if defaults is not None and not isinstance(defaults, dict):
        raise ConfigPipelineError("'defaults' must be a mapping when provided")

    pipelines = config.get("pipelines")
    if not isinstance(pipelines, dict) or not pipelines:
        raise ConfigPipelineError("Config must contain at least one pipeline")

    default_pipeline = config.get("default_pipeline")
    if default_pipeline is not None:
        if not isinstance(default_pipeline, str) or not default_pipeline:
            raise ConfigPipelineError("default_pipeline must be a non-empty string")
        if default_pipeline not in pipelines:
            raise ConfigPipelineError(f"default_pipeline references unknown pipeline '{default_pipeline}'")

    for pipeline_name, pipeline in pipelines.items():
        if not isinstance(pipeline, dict):
            raise ConfigPipelineError(f"Pipeline '{pipeline_name}' must be a mapping")
        inputs = pipeline.get("inputs", {})
        if inputs is not None and not isinstance(inputs, dict):
            raise ConfigPipelineError(f"Pipeline '{pipeline_name}' inputs must be a mapping")
        steps = pipeline.get("steps")
        if not isinstance(steps, list) or not steps:
            raise ConfigPipelineError(f"Pipeline '{pipeline_name}' must contain non-empty steps")
        seen_ids: set[str] = set()
        for index, step in enumerate(steps, start=1):
            if not isinstance(step, dict):
                raise ConfigPipelineError(f"Pipeline '{pipeline_name}' step {index} must be a mapping")
            step_id = step.get("id")
            if not isinstance(step_id, str) or not step_id:
                raise ConfigPipelineError(f"Pipeline '{pipeline_name}' step {index} requires a non-empty id")
            if step_id in seen_ids:
                raise ConfigPipelineError(f"Pipeline '{pipeline_name}' has duplicate step id '{step_id}'")
            seen_ids.add(step_id)

            selectors = [key for key in ("op", "method", "use") if key in step]
            if len(selectors) != 1:
                raise ConfigPipelineError(
                    f"Pipeline '{pipeline_name}' step '{step_id}' must set exactly one of op, method, or use"
                )
            if "use" in step and step["use"] not in pipelines:
                raise ConfigPipelineError(
                    f"Pipeline '{pipeline_name}' step '{step_id}' references unknown pipeline '{step['use']}'"
                )
            for mapping_key in ("params", "inputs", "sources"):
                if mapping_key in step and not isinstance(step[mapping_key], dict):
                    raise ConfigPipelineError(
                        f"Pipeline '{pipeline_name}' step '{step_id}' field '{mapping_key}' must be a mapping"
                    )

    _validate_no_cycles(pipelines)


def run_config_pipeline(
    config_path: str | Path,
    *,
    pipeline: str | None = None,
    inputs: Mapping[str, str | Path] | None = None,
    outdir: str | Path | None = None,
    dry_run: bool = False,
    resume: bool | None = None,
    overwrite: bool | None = None,
) -> PipelineRunResult:
    """Load, validate, and run one pipeline from a YAML config file."""

    config = load_pipeline_config(config_path)
    validate_pipeline_config(config)
    runner = ConfigPipelineRunner(
        config,
        config_path=Path(config_path),
        outdir=outdir,
        dry_run=dry_run,
        resume=resume,
        overwrite=overwrite,
    )
    return runner.run(pipeline=pipeline, inputs=inputs or {})


class ConfigPipelineRunner:
    """Executor for version-1 YAML pipeline configs."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        config_path: Path | None = None,
        outdir: str | Path | None = None,
        dry_run: bool = False,
        resume: bool | None = None,
        overwrite: bool | None = None,
    ) -> None:
        validate_pipeline_config(config)
        self.config = dict(config)
        self.config_path = config_path
        self.defaults = dict(self.config.get("defaults") or {})
        self.outdir = Path(outdir or self.defaults.get("outdir") or "outputs/pipelines")
        self.dry_run = dry_run
        self.resume = bool(self.defaults.get("resume", False)) if resume is None else resume
        self.overwrite = bool(self.defaults.get("overwrite", False)) if overwrite is None else overwrite
        if self.resume and self.overwrite:
            raise ConfigPipelineError("resume and overwrite cannot both be enabled")
        self.records: list[StepRecord] = []

    @property
    def pipelines(self) -> dict[str, Any]:
        return self.config["pipelines"]

    def run(
        self,
        *,
        pipeline: str | None = None,
        inputs: Mapping[str, str | Path] | None = None,
    ) -> PipelineRunResult:
        pipeline_name = self._select_pipeline(pipeline)
        output = self._run_pipeline(
            pipeline_name,
            input_overrides=dict(inputs or {}),
            run_dir=self.outdir / pipeline_name,
            parent_context={},
        )
        return PipelineRunResult(
            pipeline=pipeline_name,
            output=str(output) if output is not None else None,
            records=tuple(self.records),
        )

    def _run_pipeline(
        self,
        pipeline_name: str,
        *,
        input_overrides: Mapping[str, str | Path],
        run_dir: Path,
        parent_context: Mapping[str, Path],
    ) -> Path | None:
        pipeline = self.pipelines[pipeline_name]
        context = self._initial_context(pipeline_name, pipeline, input_overrides, parent_context)
        last_output: Path | None = None

        for step in pipeline["steps"]:
            step_id = step["id"]
            step_dir = run_dir / step_id
            if "use" in step:
                child_inputs = {
                    name: self._resolve_reference(value, context, last_output)
                    for name, value in dict(step.get("inputs") or {}).items()
                }
                child_output = self._run_pipeline(
                    str(step["use"]),
                    input_overrides=child_inputs,
                    run_dir=step_dir,
                    parent_context=context,
                )
                if child_output is None:
                    raise ConfigPipelineError(f"Nested pipeline step '{step_id}' produced no output")
                context[step_id] = child_output
                last_output = child_output
                self.records.append(
                    StepRecord(
                        pipeline=pipeline_name,
                        step_id=step_id,
                        kind="use",
                        action="planned" if self.dry_run else "completed",
                        output=str(child_output),
                    )
                )
                continue

            output_path = self._default_output_path(step, step_dir)
            source_refs = self._resolve_step_sources(step, context, last_output)
            fingerprint = self._fingerprint_step(step, source_refs, output_path)
            action = self._prepare_output(step_id, output_path, fingerprint)

            if self.dry_run:
                graph_path = self._graph_path(step, step_dir)
                self.records.append(
                    StepRecord(
                        pipeline=pipeline_name,
                        step_id=step_id,
                        kind="op" if "op" in step else "method",
                        action=action,
                        output=str(output_path),
                        graph=str(graph_path) if "op" in step else None,
                        source_refs={name: str(value) for name, value in source_refs.items()},
                    )
                )
                context[step_id] = output_path
                last_output = output_path
                continue

            if action == "resume":
                context[step_id] = output_path
                last_output = output_path
                self.records.append(
                    StepRecord(
                        pipeline=pipeline_name,
                        step_id=step_id,
                        kind="op" if "op" in step else "method",
                        action=action,
                        output=str(output_path),
                        source_refs={name: str(value) for name, value in source_refs.items()},
                    )
                )
                continue

            step_dir.mkdir(parents=True, exist_ok=True)
            if "op" in step:
                output = self._run_operator_step(step, source_refs, output_path, step_dir)
            else:
                output = self._run_method_step(step, source_refs, output_path, step_dir)

            if output is None:
                raise ConfigPipelineError(f"Step '{step_id}' did not return an output path")
            output_path = Path(output)
            self._write_manifest(step, source_refs, output_path, fingerprint)
            context[step_id] = output_path
            last_output = output_path
            self.records.append(
                StepRecord(
                    pipeline=pipeline_name,
                    step_id=step_id,
                    kind="op" if "op" in step else "method",
                    action=action,
                    output=str(output_path),
                    graph=str(self._graph_path(step, step_dir)) if "op" in step else None,
                    source_refs={name: str(value) for name, value in source_refs.items()},
                )
            )

        return last_output

    def _select_pipeline(self, pipeline: str | None) -> str:
        if pipeline:
            if pipeline not in self.pipelines:
                raise ConfigPipelineError(f"Unknown pipeline '{pipeline}'")
            return pipeline
        default_pipeline = self.config.get("default_pipeline")
        if default_pipeline:
            return str(default_pipeline)
        names = list(self.pipelines)
        if len(names) != 1:
            raise ConfigPipelineError("Config defines multiple pipelines; pass --pipeline")
        return names[0]

    def _initial_context(
        self,
        pipeline_name: str,
        pipeline: Mapping[str, Any],
        input_overrides: Mapping[str, str | Path],
        parent_context: Mapping[str, Path],
    ) -> dict[str, Path]:
        declared_inputs = dict(pipeline.get("inputs") or {})
        context = dict(parent_context)
        for name, default_value in declared_inputs.items():
            value = input_overrides.get(name, default_value)
            if value is None:
                raise ConfigPipelineError(f"Pipeline '{pipeline_name}' requires input '{name}'")
            context[name] = self._resolve_reference(value, context, None)
        for name, value in input_overrides.items():
            if name not in context:
                context[name] = self._resolve_reference(value, context, None)
        return context

    def _resolve_reference(
        self,
        value: str | Path,
        context: Mapping[str, Path],
        last_output: Path | None,
    ) -> Path:
        if isinstance(value, Path):
            return value
        if value == "$previous":
            if last_output is None:
                raise ConfigPipelineError("$previous cannot be used before a step has produced output")
            return last_output
        if isinstance(value, str) and value in context:
            return context[value]
        return Path(value)

    def _resolve_step_sources(
        self,
        step: Mapping[str, Any],
        context: Mapping[str, Path],
        last_output: Path | None,
    ) -> dict[str, Path]:
        if "sources" in step:
            return {
                str(name): self._resolve_reference(value, context, last_output)
                for name, value in dict(step["sources"]).items()
            }
        if "source" in step:
            return {"source": self._resolve_reference(step["source"], context, last_output)}
        if last_output is not None:
            return {"source": last_output}
        declared_inputs = [value for value in context.items()]
        if len(declared_inputs) == 1:
            name, value = declared_inputs[0]
            return {name: value}
        raise ConfigPipelineError(f"Step '{step['id']}' needs explicit source or sources")

    def _default_output_path(self, step: Mapping[str, Any], step_dir: Path) -> Path:
        output = step.get("output")
        if output:
            return Path(output)
        fmt = str(step.get("format") or self.defaults.get("format") or "BEAM-DIMAP")
        return step_dir / f"{step['id']}{GPT.EXTENSIONS_MAP.get(fmt, '.dim')}"

    def _graph_path(self, step: Mapping[str, Any], step_dir: Path) -> Path:
        return step_dir / f"{step['id']}_graph.xml"

    def _prepare_output(self, step_id: str, output_path: Path, fingerprint: str) -> str:
        if not output_path.exists():
            return "planned" if self.dry_run else "run"
        if self.resume:
            manifest = self._manifest_path(output_path)
            if manifest.exists():
                data = json.loads(manifest.read_text(encoding="utf-8"))
                if data.get("fingerprint") == fingerprint:
                    return "resume"
            raise ConfigPipelineError(f"Existing output for step '{step_id}' does not match resume manifest")
        if self.overwrite:
            if not self.dry_run:
                self._remove_output(output_path)
            return "overwrite"
        raise ConfigPipelineError(f"Output already exists for step '{step_id}': {output_path}")

    def _run_operator_step(
        self,
        step: Mapping[str, Any],
        source_refs: Mapping[str, Path],
        output_path: Path,
        step_dir: Path,
    ) -> str | None:
        graph_path = self._graph_path(step, step_dir)
        fmt = str(step.get("format") or self.defaults.get("format") or "BEAM-DIMAP")
        self._render_operator_graph(
            graph_path=graph_path,
            graph_id=str(step["id"]),
            operator=str(step["op"]),
            source_refs=source_refs,
            params=dict(step.get("params") or {}),
        )
        gpt = self._build_gpt(next(iter(source_refs.values())), step_dir, fmt)
        parameters = {name: value for name, value in source_refs.items()}
        parameters.update({"target": output_path, "format": fmt})
        return gpt.run_graph(
            graph_path=graph_path,
            output_path=output_path,
            delete_graph=not bool(self.defaults.get("keep_graphs", True)),
            parameters=parameters,
        )

    def _run_method_step(
        self,
        step: Mapping[str, Any],
        source_refs: Mapping[str, Path],
        output_path: Path,
        step_dir: Path,
    ) -> str | None:
        fmt = str(step.get("format") or self.defaults.get("format") or "BEAM-DIMAP")
        gpt = self._build_gpt(next(iter(source_refs.values())), step_dir, fmt)
        method = self._resolve_gpt_method(gpt, str(step["method"]))
        kwargs = dict(step.get("params") or {})
        if "sources" in step:
            kwargs.update(source_refs)
        if "output_name" in inspect.signature(method).parameters and "output_name" not in kwargs:
            kwargs["output_name"] = output_path.stem
        return method(**kwargs)

    def _build_gpt(self, product: Path, outdir: Path, fmt: str) -> GPT:
        return GPT(
            product=product,
            outdir=outdir,
            format=fmt,
            gpt_path=self.defaults.get("gpt_path"),
            memory=self.defaults.get("memory"),
            parallelism=self.defaults.get("parallelism"),
            snap_userdir=self.defaults.get("snap_userdir"),
            cache_size=self.defaults.get("cache_size"),
            timeout=self.defaults.get("timeout"),
            mode=self.defaults.get("mode"),
        )

    def _resolve_gpt_method(self, gpt: GPT, method_name: str) -> Any:
        candidates = [
            method_name,
            method_name.replace("-", "_"),
            method_name.replace("-", "_").lower(),
        ]
        for candidate in candidates:
            method = getattr(gpt, candidate, None)
            if callable(method):
                return method
        raise ConfigPipelineError(f"GPT has no callable method '{method_name}'")

    def _render_operator_graph(
        self,
        *,
        graph_path: Path,
        graph_id: str,
        operator: str,
        source_refs: Mapping[str, Path],
        params: Mapping[str, Any],
    ) -> None:
        graph = ET.Element("graph", id=graph_id)
        version = ET.SubElement(graph, "version")
        version.text = "1.0"

        read_ids: dict[str, str] = {}
        for index, source_name in enumerate(source_refs, start=1):
            read_id = f"Read{index}"
            read_ids[source_name] = read_id
            node = ET.SubElement(graph, "node", id=read_id)
            ET.SubElement(node, "operator").text = "Read"
            node_params = ET.SubElement(node, "parameters")
            ET.SubElement(node_params, "file").text = "${" + source_name + "}"

        op_node = ET.SubElement(graph, "node", id="Op")
        ET.SubElement(op_node, "operator").text = operator
        sources = ET.SubElement(op_node, "sources")
        if len(read_ids) == 1:
            ET.SubElement(sources, "sourceProduct", refid=next(iter(read_ids.values())))
        else:
            for source_name, read_id in read_ids.items():
                ET.SubElement(sources, source_name, refid=read_id)
        op_params = ET.SubElement(op_node, "parameters")
        for name, value in params.items():
            ET.SubElement(op_params, str(name)).text = _render_xml_value(value)

        write_node = ET.SubElement(graph, "node", id="Write")
        ET.SubElement(write_node, "operator").text = "Write"
        write_sources = ET.SubElement(write_node, "sources")
        ET.SubElement(write_sources, "sourceProduct", refid="Op")
        write_params = ET.SubElement(write_node, "parameters")
        ET.SubElement(write_params, "file").text = "${target}"
        ET.SubElement(write_params, "formatName").text = "${format}"

        _indent_xml(graph)
        ET.ElementTree(graph).write(graph_path, encoding="utf-8", xml_declaration=True)

    def _fingerprint_step(
        self,
        step: Mapping[str, Any],
        source_refs: Mapping[str, Path],
        output_path: Path,
    ) -> str:
        payload = {
            "step": step,
            "sources": {name: str(path) for name, path in source_refs.items()},
            "output": str(output_path),
            "defaults": self.defaults,
        }
        encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _write_manifest(
        self,
        step: Mapping[str, Any],
        source_refs: Mapping[str, Path],
        output_path: Path,
        fingerprint: str,
    ) -> None:
        manifest = {
            "fingerprint": fingerprint,
            "step": step,
            "sources": {name: str(path) for name, path in source_refs.items()},
            "output": str(output_path),
        }
        self._manifest_path(output_path).write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

    def _manifest_path(self, output_path: Path) -> Path:
        return output_path.with_suffix(output_path.suffix + ".sarpyx-pipeline.json")

    def _remove_output(self, output_path: Path) -> None:
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink(missing_ok=True)
        if output_path.suffix == ".dim":
            data_dir = output_path.with_suffix("").with_name(output_path.stem + ".data")
            if data_dir.exists():
                shutil.rmtree(data_dir)
        self._manifest_path(output_path).unlink(missing_ok=True)


def _validate_no_cycles(pipelines: Mapping[str, Any]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str, chain: Iterable[str]) -> None:
        if name in visited:
            return
        if name in visiting:
            rendered = " -> ".join([*chain, name])
            raise ConfigPipelineError(f"Pipeline composition cycle detected: {rendered}")
        visiting.add(name)
        for step in pipelines[name].get("steps", []):
            if "use" in step:
                visit(str(step["use"]), [*chain, name])
        visiting.remove(name)
        visited.add(name)

    for pipeline_name in pipelines:
        visit(str(pipeline_name), [])


def _render_xml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return ",".join(_render_xml_value(item) for item in value)
    if value is None:
        return ""
    return str(value)


def _indent_xml(elem: ET.Element, level: int = 0) -> None:
    indent = "\n" + level * "    "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + "    "
        for child in elem:
            _indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = indent
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = indent
