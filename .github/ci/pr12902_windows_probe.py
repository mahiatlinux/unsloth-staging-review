from __future__ import annotations

import platform
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "studio"))

from install_sd_cpp_prebuilt import resolve_release_asset


system = platform.system()
machine = platform.machine()
assert system == "Windows", (system, machine)
assert machine.lower() in {"amd64", "x86_64"}, (system, machine)

cpu = "sd-master-813-bfbef5b-u1d02858-bin-win-cpu-x64.zip"
vulkan = "sd-master-813-bfbef5b-u1d02858-bin-win-vulkan-x64.zip"

for accelerator in ("auto", "cpu"):
    for assets in ([cpu, vulkan], [vulkan, cpu]):
        selected = resolve_release_asset(
            assets,
            system=system,
            machine=machine,
            accelerator=accelerator,
        )
        assert selected == cpu, (accelerator, assets, selected)

    for marker in ("cuda12", "vulkan", "rocm", "sycl", "musa"):
        gpu = f"sd-master-test-bin-win-{marker}-avx2-x64.zip"
        selected = resolve_release_asset(
            [gpu],
            system=system,
            machine=machine,
            accelerator=accelerator,
        )
        assert selected is None, (accelerator, gpu, selected)

    upstream_avx2 = "sd-master-test-bin-win-avx2-x64.zip"
    selected = resolve_release_asset(
        [cpu, upstream_avx2],
        system=system,
        machine=machine,
        accelerator=accelerator,
    )
    assert selected == upstream_avx2, (accelerator, selected)

print(f"PASS native Windows resolver probe on {system} {machine}")
