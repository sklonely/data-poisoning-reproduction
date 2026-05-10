import json
import math
import shutil
import time
import uuid
from pathlib import Path

def _stringify(value):
    if value is None:
        return 'null'
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value)
    return str(value)


def _json_ready(value):
    try:
        import numpy as np
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return str(value)
    return value


class LocalExperiment:
    def __init__(self, project_name, root='runs', workspace='local'):
        self.project_name = project_name
        self.root = Path(root).resolve()
        self.workspace = workspace
        self.key = uuid.uuid4().hex
        self.dir = self.root / workspace / project_name / self.key
        self.assets_dir = self.dir / 'assets'
        self.figures_dir = self.dir / 'figures'
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.dir / 'meta.json'
        self.metrics_path = self.dir / 'metrics.jsonl'
        self.params = {}
        self.others = {}
        self.tags = []
        self.name = None
        self.asset_records = []
        self.start_time_ms = int(time.time() * 1000)
        self.end_time_ms = None
        self._write_meta()

    @property
    def url(self):
        return str(self.dir)

    @property
    def id(self):
        return self.key

    def _get_experiment_url(self):
        return self.url

    def _write_meta(self):
        payload = {
            'workspace': self.workspace,
            'project_name': self.project_name,
            'key': self.key,
            'name': self.name,
            'params': self.params,
            'others': self.others,
            'tags': self.tags,
            'assets': self.asset_records,
            'start_time_ms': self.start_time_ms,
            'end_time_ms': self.end_time_ms,
            'duration_ms': None if self.end_time_ms is None else self.end_time_ms - self.start_time_ms,
        }
        self.meta_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')

    def add_tag(self, tag):
        if tag and tag not in self.tags:
            self.tags.append(tag)
            self._write_meta()

    def set_name(self, name):
        self.name = name
        self._write_meta()

    def log_parameters(self, params):
        for key, value in params.items():
            self.params[key] = _json_ready(value)
        self._write_meta()

    def log_other(self, key, value):
        self.others[key] = _json_ready(value)
        self._write_meta()

    def get_parameter(self, key):
        return self.params.get(key)

    def get_key(self):
        return self.key

    def log_asset(self, src_path, file_name=None, step=None):
        src = Path(src_path)
        stored_name = file_name or src.name
        asset_id = f'{len(self.asset_records):05d}_{stored_name}'
        dst = self.assets_dir / asset_id
        shutil.copy2(src, dst)
        self.asset_records.append({
            'assetId': asset_id,
            'fileName': stored_name,
            'step': step,
            'path': str(dst),
        })
        self._write_meta()

    def log_figure(self, figure_name=None, step=None):
        import matplotlib.pyplot as plt

        figure_name = figure_name or f'figure-{len(list(self.figures_dir.glob("*.png")))}'
        safe_name = figure_name.replace('/', '_')
        filename = f'{safe_name}-step-{step if step is not None else "na"}.png'
        plt.savefig(self.figures_dir / filename, bbox_inches='tight')

    def log_metric(self, key, value, step=None):
        self.log_metrics({key: value}, step=step)

    def log_metrics(self, metrics, step=None):
        record = {
            'step': step,
            'time_ms': int(time.time() * 1000),
            'metrics': _json_ready(metrics),
        }
        with self.metrics_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, sort_keys=True) + '\n')

    def send_notification(self, *_args, **_kwargs):
        return None

    def end(self):
        self.end_time_ms = int(time.time() * 1000)
        self._write_meta()


class LocalExperimentView:
    def __init__(self, directory):
        self.dir = Path(directory)
        self.meta = json.loads((self.dir / 'meta.json').read_text(encoding='utf-8'))
        self.url = str(self.dir)
        self.id = self.meta['key']
        self.end_server_timestamp = self.meta.get('end_time_ms') or self.meta.get('start_time_ms')

    def _get_experiment_url(self):
        return self.url

    def get_asset_list(self):
        return [
            {'assetId': asset['assetId'], 'fileName': asset['fileName'], 'step': asset.get('step')}
            for asset in self.meta.get('assets', [])
        ]

    def get_asset(self, asset_id):
        for asset in self.meta.get('assets', []):
            if asset['assetId'] == asset_id:
                return Path(asset['path']).read_bytes()
        raise KeyError(f'Asset {asset_id} not found in {self.dir}')

    def get_parameters_summary(self, key=None):
        params = self.meta.get('params', {})
        if key is not None:
            return {'valueCurrent': _stringify(params[key])}
        return [{'name': name, 'valueCurrent': _stringify(value)} for name, value in params.items()]

    def get_others_summary(self, key):
        value = self.meta.get('others', {}).get(key)
        return [] if value is None else [value]


class LocalAPI:
    def __init__(self, root='runs', workspace='local'):
        self.root = Path(root).resolve()
        self.workspace = workspace

    def _project_dir(self, project_name):
        return self.root / self.workspace / project_name

    def get_experiment(self, workspace, project_name, key):
        exp_dir = self.root / workspace / project_name / key
        return LocalExperimentView(exp_dir)

    def get(self, *parts):
        if len(parts) == 1:
            project_ref = parts[0]
            workspace, project_name = project_ref.split('/', 1)
            project_dir = self.root / workspace / project_name
            return [LocalExperimentView(path) for path in sorted(project_dir.iterdir()) if path.is_dir()]
        if len(parts) == 3:
            workspace, project_name, key = parts
            return self.get_experiment(workspace, project_name, key)
        raise ValueError(f'Unsupported LocalAPI.get signature: {parts}')

    def find_by_uid(self, project_name, uid):
        project_dir = self._project_dir(project_name)
        matches = []
        if not project_dir.exists():
            return matches
        for path in sorted(project_dir.iterdir()):
            if not path.is_dir():
                continue
            expt = LocalExperimentView(path)
            if str(expt.meta.get('params', {}).get('uid')) == str(uid):
                matches.append(expt)
        return matches
