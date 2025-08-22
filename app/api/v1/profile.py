from fastapi import APIRouter, HTTPException, Path
from fastapi.responses import StreamingResponse
import pandas as pd
import io

from controllers.Module import profile_to_excel

router = APIRouter()

@router.get("/{scenario_id}/excel", response_class=StreamingResponse)
def export_profile_excel(
    scenario_id: int = Path(..., ge=1, description="SIM_SCENARIOS.scenario_id")
):
    try:
        df = profile_to_excel(scenario_id)
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail="No profile data.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"build_profile_df failed: {e}")

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False)
    buf.seek(0)

    filename = f"scenario_{scenario_id}_profile.xlsx"
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )
