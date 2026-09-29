from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

from .models import PersonaProfile
from .personas import PHYSICAL


def _find_binary(name: str) -> str:
    exe = shutil.which(name)
    if exe:
        return exe
    raise FileNotFoundError(
        f"{name} not found. Install Eclipse SUMO and put its bin directory on PATH."
    )


def write_demo_network(workdir: str | Path) -> Path:
    """Create a junction with one direct turn and one valid detour."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    nod = workdir / "decision_lab.nod.xml"
    edg = workdir / "decision_lab.edg.xml"
    con = workdir / "decision_lab.con.xml"
    net = workdir / "decision_lab.net.xml"

    nod.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<nodes>
  <node id="W" x="-220" y="0" type="priority"/>
  <node id="J" x="0" y="0" type="priority"/>
  <node id="E" x="150" y="0" type="priority"/>
  <node id="EO" x="320" y="0" type="priority"/>
  <node id="NE" x="150" y="150" type="priority"/>
  <node id="NM" x="0" y="150" type="priority"/>
  <node id="N" x="0" y="320" type="priority"/>
</nodes>
""",
        encoding="utf-8",
    )
    edg.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<edges>
  <edge id="WJ" from="W" to="J" numLanes="1" speed="13.9"/>
  <edge id="JN" from="J" to="NM" numLanes="1" speed="11.1"/>
  <edge id="NS" from="NM" to="N" numLanes="1" speed="13.9"/>
  <edge id="JE" from="J" to="E" numLanes="1" speed="11.1"/>
  <edge id="EE" from="E" to="EO" numLanes="1" speed="13.9"/>
  <edge id="EN" from="E" to="NE" numLanes="1" speed="11.1"/>
  <edge id="N2" from="NE" to="NM" numLanes="1" speed="11.1"/>
</edges>
""",
        encoding="utf-8",
    )
    con.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<connections>
  <connection from="WJ" to="JN" fromLane="0" toLane="0"/>
  <connection from="JN" to="NS" fromLane="0" toLane="0"/>
  <connection from="WJ" to="JE" fromLane="0" toLane="0"/>
  <connection from="JE" to="EE" fromLane="0" toLane="0"/>
  <connection from="JE" to="EN" fromLane="0" toLane="0"/>
  <connection from="EN" to="N2" fromLane="0" toLane="0"/>
  <connection from="N2" to="NS" fromLane="0" toLane="0"/>
</connections>
""",
        encoding="utf-8",
    )

    subprocess.run(
        [
            _find_binary("netconvert"),
            "--node-files", str(nod),
            "--edge-files", str(edg),
            "--connection-files", str(con),
            "--output-file", str(net),
            "--no-turnarounds", "true",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return net


def write_vtypes(
    workdir: str | Path,
    profiles: list[PersonaProfile],
) -> Path:
    workdir = Path(workdir)
    path = workdir / "decision_lab_types.add.xml"
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        "<additional>",
        "  <!-- Phase-2 heterogeneity; longitudinal tau stays frozen at H0=0.50 s. -->",
    ]
    for p in profiles:
        phys = PHYSICAL[p.vehicle_class]
        lines.append(
            f'  <vType id="{p.type_id}" '
            f'vClass="{phys["vclass"]}" '
            f'length="{phys["length_m"]:.3f}" '
            f'width="{phys["width_m"]:.3f}" '
            f'carFollowModel="Krauss" laneChangeModel="SL2015" '
            f'tau="{p.tau_s:.3f}" minGap="0.50" '
            f'minGapLat="{p.min_gap_lat_m:.3f}" '
            f'lcSublane="1.0" lcPushy="0.0" '
            f'speedFactor="{p.speed_factor:.4f}" speedDev="0.0"/>'
        )
    lines.append("</additional>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
