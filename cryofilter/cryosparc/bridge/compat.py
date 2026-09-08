"""Compatibility wrapper for CryoSPARC Tools version differences."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable


def find_project(client: object, project_uid: str):
    finder = getattr(client, "find_project", None)
    if callable(finder):
        return finder(project_uid)
    projects = getattr(client, "projects", None)
    finder = getattr(projects, "find_one", None)
    if callable(finder):
        return finder({"uid": project_uid})
    raise AttributeError("Could not find a CryoSPARC project lookup API")


def find_job(client: object, project: object, project_uid: str, job_uid: str):
    finder = getattr(project, "find_job", None)
    if callable(finder):
        return finder(job_uid)
    finder = getattr(client, "find_job", None)
    if callable(finder):
        return finder(project_uid, job_uid)
    raise AttributeError("Could not find a CryoSPARC job lookup API")


def find_external_job(client: object, project: object, project_uid: str, job_uid: str):
    finder = getattr(project, "find_external_job", None)
    if callable(finder):
        return finder(job_uid)
    finder = getattr(client, "find_external_job", None)
    if callable(finder):
        return finder(project_uid, job_uid)
    job = find_job(client, project, project_uid, job_uid)
    if callable(getattr(job, "add_output", None)) and callable(
        getattr(job, "save_output", None)
    ):
        return job
    raise AttributeError("Could not find a CryoSPARC External Job lookup API")


def find_workspace(client: object, project: object, project_uid: str, workspace_uid: str):
    finder = getattr(project, "find_workspace", None)
    if callable(finder):
        return finder(workspace_uid)
    finder = getattr(client, "find_workspace", None)
    if callable(finder):
        try:
            return finder(project_uid, workspace_uid)
        except TypeError:
            return finder(workspace_uid)
    return None


def create_external_job(project: object, workspace_uid: str, title: str, desc: str | None = None):
    creator = getattr(project, "create_external_job")
    try:
        return creator(workspace_uid=workspace_uid, title=title, desc=desc)
    except TypeError:
        if desc is None:
            return creator(workspace_uid, title)
        return creator(workspace_uid, title, desc)


def connect_input(
    external_job: object,
    *,
    type_name: str,
    input_name: str,
    source_job_uid: str,
    source_output_name: str,
    slots: Iterable[str] = (),
) -> None:
    slot_list = list(slots)
    add_input = getattr(external_job, "add_input")
    try:
        add_input(type=type_name, name=input_name, slots=slot_list)
    except TypeError:
        add_input(type_name, input_name, slots=slot_list)

    connect = getattr(external_job, "connect")
    try:
        connect(
            target_input=input_name,
            source_job_uid=source_job_uid,
            source_output=source_output_name,
            slots=slot_list,
        )
    except TypeError:
        connect(input_name, source_job_uid, source_output_name)


def connect_particle_input(
    external_job: object,
    *,
    input_name: str,
    source_job_uid: str,
    source_output_name: str,
    slots: Iterable[str] = (),
) -> None:
    connect_input(
        external_job,
        type_name="particle",
        input_name=input_name,
        source_job_uid=source_job_uid,
        source_output_name=source_output_name,
        slots=slots,
    )


def connect_exposure_input(
    external_job: object,
    *,
    input_name: str,
    source_job_uid: str,
    source_output_name: str,
    slots: Iterable[str] = (),
) -> None:
    connect_input(
        external_job,
        type_name="exposure",
        input_name=input_name,
        source_job_uid=source_job_uid,
        source_output_name=source_output_name,
        slots=slots,
    )


def start_external_job(external_job: object) -> None:
    starter = getattr(external_job, "start", None)
    if callable(starter):
        starter()


def stop_external_job(external_job: object, error: str | None = None) -> None:
    stopper = getattr(external_job, "stop", None)
    if not callable(stopper):
        return
    if error:
        try:
            stopper(error=error)
        except TypeError:
            stopper(error)
    else:
        stopper()


@contextmanager
def run_external_job(external_job: object):
    runner = getattr(external_job, "run", None)
    if callable(runner):
        with runner():
            yield
        return

    start_external_job(external_job)
    try:
        yield
    except Exception as exc:
        stop_external_job(external_job, error=str(exc))
        raise
    else:
        stop_external_job(external_job)


def add_particle_output(
    external_job: object,
    *,
    name: str,
    slots: Iterable[str],
    title: str,
    passthrough: str = "input_particles",
    alloc: object | None = None,
) -> object:
    add_output = getattr(external_job, "add_output")
    slot_list = list(slots)
    try:
        return add_output(
            type="particle",
            name=name,
            passthrough=passthrough,
            slots=slot_list,
            title=title,
            alloc=alloc,
        )
    except TypeError:
        try:
            return add_output(
                "particle",
                name,
                passthrough=passthrough,
                slots=slot_list,
                title=title,
                alloc=alloc,
            )
        except TypeError:
            return add_output("particle", name, slot_list, title)


def save_output(external_job: object, name: str, dataset: object) -> None:
    saver = getattr(external_job, "save_output")
    try:
        saver(name=name, dataset=dataset)
    except TypeError:
        saver(name, dataset)


def log_plot(
    external_job: object,
    *,
    figure: object,
    text: str,
    formats: Iterable[str] = ("png",),
    raw_data_file: object | None = None,
    raw_data_format: str | None = None,
) -> str | None:
    logger = getattr(external_job, "log_plot", None)
    if not callable(logger):
        return None
    kwargs = {
        "figure": figure,
        "text": text,
        "formats": list(formats),
    }
    if raw_data_file is not None:
        kwargs["raw_data_file"] = raw_data_file
    if raw_data_format is not None:
        kwargs["raw_data_format"] = raw_data_format
    try:
        return logger(**kwargs)
    except TypeError:
        return logger(figure, text, formats=list(formats))


def object_uid(obj: object) -> str | None:
    value = getattr(obj, "uid", None)
    if value:
        return str(value)
    doc = getattr(obj, "doc", None)
    if isinstance(doc, dict) and doc.get("uid"):
        return str(doc["uid"])
    model = getattr(obj, "model", None)
    uid = getattr(model, "uid", None)
    return None if uid is None else str(uid)


def dataset_prefixes(dataset: object) -> list[str]:
    prefixes = getattr(dataset, "prefixes", None)
    if callable(prefixes):
        return [str(value) for value in prefixes()]
    dtype = getattr(dataset, "dtype", None)
    names = getattr(dtype, "names", None) or ()
    return sorted({str(name).split("/", 1)[0] for name in names if "/" in str(name)})


def dataset_uids(dataset: object):
    import numpy as np

    dtype = getattr(dataset, "dtype", None)
    names = getattr(dtype, "names", None) or ()
    if "uid" not in names:
        fields = getattr(dataset, "fields", None)
        if callable(fields) and "uid" not in fields():
            raise ValueError("CryoSPARC particle dataset does not contain a uid field")
    try:
        return np.asarray(dataset["uid"])
    except Exception:
        records = getattr(dataset, "to_records", None)
        if callable(records):
            return np.asarray(records()["uid"])
        raise ValueError("CryoSPARC particle dataset does not contain a readable uid field")


def dataset_take(dataset: object, indices: object):
    take = getattr(dataset, "take", None)
    if callable(take):
        return take(indices)
    return dataset[indices]


def dataset_filter_prefixes(dataset: object, prefixes: list[str]):
    filter_prefixes = getattr(dataset, "filter_prefixes", None)
    if callable(filter_prefixes):
        return filter_prefixes(prefixes, copy=True)
    filter_fields = getattr(dataset, "filter_fields", None)
    if callable(filter_fields):
        return filter_fields(
            lambda name: str(name) == "uid" or str(name).split("/", 1)[0] in prefixes,
            copy=True,
        )
    return dataset


def supports_job_plot_logs() -> bool:
    try:
        from cryosparc.controllers.job import JobController

        return callable(getattr(JobController, "log_plot", None))
    except Exception:
        return False


__all__ = [
    "add_particle_output",
    "connect_exposure_input",
    "connect_input",
    "connect_particle_input",
    "create_external_job",
    "dataset_filter_prefixes",
    "dataset_prefixes",
    "dataset_take",
    "dataset_uids",
    "find_external_job",
    "find_job",
    "find_project",
    "find_workspace",
    "log_plot",
    "object_uid",
    "run_external_job",
    "save_output",
    "start_external_job",
    "stop_external_job",
    "supports_job_plot_logs",
]
