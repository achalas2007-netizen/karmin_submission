
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import random
import time
import re
import math

st.set_page_config(page_title="KARMIN Autonomous Engine", layout="wide")

if 'sovereign_log' not in st.session_state:
    st.session_state.sovereign_log = []
if 'sweep_done' not in st.session_state:
    st.session_state.sweep_done = False
if 'total_saved' not in st.session_state:
    st.session_state.total_saved = 0.0
if 'undo_cache' not in st.session_state:
    st.session_state.undo_cache = []
# Step 4C session state — sovereign engine
if 'engine_log' not in st.session_state:
    st.session_state.engine_log = []
if 'engine_done' not in st.session_state:
    st.session_state.engine_done = False
if 'engine_monthly_recovery' not in st.session_state:
    st.session_state.engine_monthly_recovery = 0.0
if 'engine_arr' not in st.session_state:
    st.session_state.engine_arr = 0.0

#step 1:UNIVERSAL INGESTOR  
class KarminUniversalServiceIngestor:
    def __init__(self, provider_name="AWS", linked_account_id=None):
        self.provider = provider_name
        self.linked_account_id = linked_account_id

    def normalize_service_matrix(self, raw_payload_df):
        timestamp = datetime.now(timezone.utc).isoformat()
        grouped = raw_payload_df.groupby("ServiceName", as_index=False).agg({
            "Cost_USD":      "sum",
            "CPU_Percent":   "mean",
            "Traffic":       "sum",
            "Actual_Val":    "mean",
            "Provisioned_Cap": "mean",
            "Dependency":    "max",
            "Instance_ID":   "first"
        })
        grouped["Timestamp"]      = timestamp
        grouped["Provider"]       = self.provider
        grouped["LinkedAccountID"] = self.linked_account_id
        return grouped

# step 2: CONTEXT PROFILER 
class KarminContextProfiler:
    def __init__(self):
        self.threshold_config = {
            "AmazonEC2": {"metric": "CPU_Percent",      "healthy_floor_pct": 30.0, "critical_floor_pct": 10.0},
            "AmazonRDS": {"metric": "Connections",      "healthy_floor_pct": 40.0, "critical_floor_pct": 15.0},
            "AmazonS3":  {"metric": "Access_Frequency", "healthy_floor_pct": 20.0, "critical_floor_pct":  5.0}
        }

    def get_service_config(self, service):
        return self.threshold_config.get(service, {
            "metric": "Generic_Utilization",
            "healthy_floor_pct": 25.0,
            "critical_floor_pct": 10.0
        })

    def compute_utilization_pct(self, actual_val, provisioned_cap):
        if provisioned_cap <= 0:
            return 0.0
        return max(0.0, min((actual_val / provisioned_cap) * 100, 100.0))

    def compute_inefficiency_pct(self, utilization_pct, healthy_floor_pct):
        if utilization_pct >= healthy_floor_pct:
            return 0.0
        return ((healthy_floor_pct - utilization_pct) / healthy_floor_pct) * 100.0

    def compute_potential_waste_usd(self, cost_usd, inefficiency_pct):
        return cost_usd * (inefficiency_pct / 100.0)

    def compute_uer(self, utilization_pct, cost_usd):
        if cost_usd <= 0:
            return 0.0
        return utilization_pct / cost_usd

    def classify_status(self, utilization_pct, healthy_floor_pct, critical_floor_pct):
        if utilization_pct < critical_floor_pct:
            return "CRITICAL_WASTE_CANDIDATE"
        elif utilization_pct < healthy_floor_pct:
            return "WATCH"
        return "HEALTHY"

    def profile_batch(self, raw_payload_df):
        results = []
        for _, row in raw_payload_df.iterrows():
            config = self.get_service_config(row["ServiceName"])
            util   = self.compute_utilization_pct(row["Actual_Val"], row["Provisioned_Cap"])
            ineff  = self.compute_inefficiency_pct(util, config["healthy_floor_pct"])
            waste  = self.compute_potential_waste_usd(row["Cost_USD"], ineff)
            uer    = self.compute_uer(util, row["Cost_USD"])
            status = self.classify_status(util, config["healthy_floor_pct"], config["critical_floor_pct"])
            results.append({
                "Timestamp":           row["Timestamp"],
                "ServiceName":         row["ServiceName"],
                "Instance_ID":         row["Instance_ID"],
                "Cost_USD":            round(row["Cost_USD"], 2),
                "CPU_Percent":         round(row["CPU_Percent"], 2),
                "Traffic":             round(row["Traffic"], 2),
                "Actual_Val":          row["Actual_Val"],
                "Provisioned_Cap":     row["Provisioned_Cap"],
                "Utilization_Pct":     round(util, 2),
                "Inefficiency_Pct":    round(ineff, 2),
                "Potential_Waste_USD": round(waste, 2),
                "UER":                 round(uer, 4),
                "Dependency":          row["Dependency"],
                "Status":              status
            })
        return pd.DataFrame(results)

#step 3: ENSEMBLE ANOMALY ENGINE
class KarminEnsembleEngine:
    def __init__(self, z_threshold=2.0, newton_threshold=1.25, tpmad_threshold=3.0):
        self.z_threshold      = z_threshold
        self.newton_threshold = newton_threshold
        self.tpmad_threshold  = tpmad_threshold

    def z_score_detector(self, values):
        baseline = values[:-1]
        mean_val = np.mean(baseline)
        std_val  = np.std(baseline)
        if std_val == 0:
            return 0.0, False
        z = (values[-1] - mean_val) / std_val
        return round(z, 4), z > self.z_threshold

    def newton_interpolation(self, x_values, y_values, x_target):
        n    = len(x_values)
        coef = np.zeros([n, n])
        coef[:, 0] = y_values
        for j in range(1, n):
            for i in range(n - 1, j - 1, -1):
                denom = x_values[i] - x_values[i - j]
                if denom == 0:
                    return float(np.mean(y_values))
                coef[i, j] = (coef[i, j - 1] - coef[i - 1, j - 1]) / denom
        prediction = coef[n - 1, n - 1]
        for i in range(n - 2, -1, -1):
            prediction = coef[i, i] + (x_target - x_values[i]) * prediction
        return float(prediction)

    def newton_detector(self, values):
        baseline  = values[:-1]
        x         = np.arange(len(baseline))
        y         = np.array(baseline, dtype=float)
        predicted = self.newton_interpolation(x, y, len(baseline))
        ratio     = values[-1] / predicted if predicted > 0 else np.inf
        return round(predicted, 4), round(ratio, 4), ratio > self.newton_threshold

    def tpmad_detector(self, values):
        baseline   = np.array(values[:-1], dtype=float)
        median_val = np.median(baseline)
        mad        = np.median(np.abs(baseline - median_val))
        if mad == 0:
            return 0.0, False
        score = abs(values[-1] - median_val) / mad
        return round(score, 4), score > self.tpmad_threshold

    def evaluate_service(self, service_history_df):
        values = service_history_df["Potential_Waste_USD"].tolist()
        if len(values) < 5:
            return None
        latest                         = service_history_df.iloc[-1]
        z_score,   z_flag              = self.z_score_detector(values)
        predicted, ratio, newton_flag  = self.newton_detector(values)
        tpmad_score, tpmad_flag        = self.tpmad_detector(values)
        votes = sum([z_flag, newton_flag, tpmad_flag])
        return {
            "ServiceName":         latest["ServiceName"],
            "Instance_ID":         latest["Instance_ID"],
            "Cost_USD":            latest["Cost_USD"],
            "CPU_Percent":         latest["CPU_Percent"],
            "Traffic":             latest["Traffic"],
            "Potential_Waste_USD": latest["Potential_Waste_USD"],
            "UER":                 latest["UER"],
            "Dependency":          latest["Dependency"],
            "ZScore":              z_score,
            "Newton_Prediction":   predicted,
            "Newton_Ratio":        ratio,
            "TPMAD_Score":         tpmad_score,
            "Z_Flag":              z_flag,
            "Newton_Flag":         newton_flag,
            "TPMAD_Flag":          tpmad_flag,
            "Vote_Count":          votes,
            "Anomaly":             votes >= 2
        }

#step 4: SOVEREIGN PHYSICS ENGINE
def get_confidence(sensors: dict) -> float:
    weights = {'z_score': 1.0, 'slope': 0.8, 'tpmad': 1.2}
    total_w = sum(weights.values())
    score   = 0.0
    for k, v in sensors.items():
        intensity = 1.0 / (1.0 + math.exp(-2.0 * (v - 2.0)))
        score    += intensity * weights.get(k, 0)
    return round(min(1.0, score / total_w), 4)

def get_dependency_risk(inbound: int, outbound: int) -> float:
    gravity = (inbound * 5.0) + (outbound * 1.0)
    try:
        risk = 1.0 / (1.0 + math.exp(-1.2 * (gravity - 2.0)))
    except OverflowError:
        risk = 1.0
    return round(min(1.0, max(0.0, risk)), 4)

def get_savings_impact(savings: float) -> float:
    if savings < 10:
        return 0.0
    return round(min(1.0, math.log10(savings) / 3.0), 4)

def get_rollback_readiness(has_snap: bool) -> float:
    return 1.0 if has_snap else 0.0

#step 5: KARMIN SOVEREIGN AGENT
class KarminSovereignAgent:
    def __init__(self):
        self.total_saved_this_pass = 0.0

    def _build_sensor_packet(self, ensemble_row: dict) -> dict:
        dep     = ensemble_row.get("Dependency", False)
        service = ensemble_row.get("ServiceName", "")
        return {
            "sensors": {
                "z_score": ensemble_row.get("ZScore", 0.0),
                "slope":   ensemble_row.get("Newton_Ratio", 0.0),
                "tpmad":   ensemble_row.get("TPMAD_Score", 0.0)
            },
            "inbound":      2 if dep else 0,
            "outbound":     1 if dep else 1,
            "savings":      round(ensemble_row.get("Potential_Waste_USD", 0.0) * 24 * 30, 2),
            "has_snapshot": False if "S3" in service else True
        }

    def evaluate_and_execute(self, ensemble_row: dict) -> dict:
        r_id = ensemble_row.get("Instance_ID", "unknown")
        data = self._build_sensor_packet(ensemble_row)

        C    = get_confidence(data["sensors"])
        Rs   = get_dependency_risk(data["inbound"], data["outbound"])
        S    = get_savings_impact(data["savings"])
        Roll = get_rollback_readiness(data["has_snapshot"])
        A    = round(C * (1.0 - Rs) * S * Roll, 4)

        if Roll == 0.0:
            status = "BLOCKED"
            reason = (
                f"No snapshot exists for `{r_id}`. "
                f"KARMIN does not execute irreversible actions. "
                f"Create a backup — this instance will auto-terminate on next sweep."
            )
        elif A > 0.5:
            status = "TERMINATED"
            self.total_saved_this_pass += data["savings"]
            reason = (
                f"Confidence **{int(C*100)}%** (Z + Newton + TPMAD consensus). "
                f"Dependency blast radius: **{int(Rs*100)}%** (isolated node). "
                f"Authority Vector A = **{A}** exceeded threshold 0.5. "
                f"Instance terminated. Rollback ready."
            )
        elif C > 0.7 and Rs > 0.4 and Roll == 1.0:
            status = "RESIZED"
            self.total_saved_this_pass += data["savings"]
            reason = (
                f"Confidence **{int(C*100)}%** but dependency risk **{int(Rs*100)}%** too high for termination. "
                f"Instance is a structural hub — downsized safely. "
                f"Authority Vector A = **{A}**."
            )
        else:
            status = "IGNORED"
            reason = (
                f"Authority Vector A = **{A}** below all action thresholds. "
                f"Confidence: {int(C*100)}%, Savings impact: {round(S,2)}, Risk: {int(Rs*100)}%. "
                f"Monitoring continues."
            )

        return {
            "id":           r_id,
            "service":      ensemble_row.get("ServiceName", ""),
            "status":       status,
            "reason":       reason,
            "savings":      data["savings"],
            "C":            C,
            "Rs":           Rs,
            "S":            S,
            "Roll":         Roll,
            "A":            A,
            "anomaly":      ensemble_row.get("Anomaly", False),
            "vote_count":   ensemble_row.get("Vote_Count", 0),
            "z_score":      ensemble_row.get("ZScore", 0.0),
            "newton_ratio": ensemble_row.get("Newton_Ratio", 0.0),
            "tpmad":        ensemble_row.get("TPMAD_Score", 0.0),
            "cpu":          ensemble_row.get("CPU_Percent", 0.0),
            "uer":          ensemble_row.get("UER", 0.0),
        }

    def run_sweep(self, ensemble_df: pd.DataFrame) -> list:
        self.total_saved_this_pass = 0.0
        log = []
        for _, row in ensemble_df.iterrows():
            if row.get("Anomaly", False):
                result = self.evaluate_and_execute(row.to_dict())
                log.append(result)
        return log


# STEP 4B-BRIDGE: ENRICHMENT FUNCTION
# Bridges Step 3 ensemble output → KarminSovereignEngine node format
# In production: Storage_GB from DescribeVolumes, RDS_Sync_Calls from
# CloudWatch, Is_Stateful from service metadata, Has_Snapshot from
# DescribeSnapshots. Mock values used here for prototype.
def bridge_from_brain(ensemble_row: dict) -> dict:
    """
    Enriches ensemble_row with infrastructure metadata required
    by KarminSovereignEngine.
    Adds: Storage_GB, RDS_Sync_Calls_Per_Sec, Is_Stateful,
          Has_Snapshot, Monthly_Waste
    """
    service = ensemble_row.get("ServiceName", "")
    dep     = ensemble_row.get("Dependency", False)

    # Mock enrichment — production values come from AWS APIs
    storage_map = {
        "AmazonEC2": random.randint(20, 100),
        "AmazonRDS": random.randint(100, 500),
        "AmazonS3":  random.randint(500, 2000),
    }
    storage_gb        = storage_map.get(service, 50)
    rds_sync_calls    = random.randint(5, 80) if "RDS" in service else random.randint(0, 10)
    is_stateful       = ("RDS" in service) or dep
    # S3 has no snapshot mechanism — safety gate will BLOCK it
    has_snapshot      = "S3" not in service
    monthly_waste     = round(ensemble_row.get("Potential_Waste_USD", 0.0) * 24 * 30, 2)

    return {
        "Instance_ID":             ensemble_row.get("Instance_ID", "unknown"),
        "ServiceName":             service,
        "CPU_Percent":             ensemble_row.get("CPU_Percent", 50.0),
        "Storage_GB":              storage_gb,
        "RDS_Sync_Calls_Per_Sec":  rds_sync_calls,
        "Is_Stateful":             is_stateful,
        "Has_Snapshot":            has_snapshot,
        "Monthly_Waste":           monthly_waste,
        "Dependency":              dep,
        # Pass through ensemble detector scores for audit trail
        "ZScore":                  ensemble_row.get("ZScore", 0.0),
        "Newton_Ratio":            ensemble_row.get("Newton_Ratio", 0.0),
        "TPMAD_Score":             ensemble_row.get("TPMAD_Score", 0.0),
        "Vote_Count":              ensemble_row.get("Vote_Count", 0),
        "Anomaly":                 ensemble_row.get("Anomaly", False),
    }


# STEP 4C: KARMIN SOVEREIGN ENGINE
# Authority Vector: A = Confidence × (1 − Risk) × Savings × Safety
# Produces: AUTO_TERMINATE / TERMINATE / DOWNSIZE / BLOCK / MONITOR
# Tracks:   Monthly Recovery + ARR Reallocated
class KarminSovereignEngine:
    def __init__(self, mode="ESG_COMPLIANCE"):
        self.mode                   = mode
        self.total_monthly_savings  = 0.0
        self.total_arr_recovered    = 0.0
        self.actions_executed       = 0

    def _get_confidence(self, utilization: float) -> float:
        """Sigmoid: High utilization → low waste confidence. Low util → high confidence."""
        return round(1.0 / (1.0 + math.exp(0.15 * (utilization - 25.0))), 2)

    def _get_risk(self, row: dict) -> float:
        """
        Three-factor blast radius:
        1. Data size risk   — log-scale storage penalty
        2. Dependency risk  — exponential RDS sync call penalty
        3. Stateful penalty — hard 1.0 if service holds state
        """
        data_size_risk   = round(min(1.0, math.log10(max(1, row.get("Storage_GB", 0))) / 3.0), 2)
        dependency_risk  = round(min(1.0, 1.0 - math.exp(-0.005 * row.get("RDS_Sync_Calls_Per_Sec", 0))), 2)
        state_penalty    = 1.0 if row.get("Is_Stateful") else 0.0
        return max(data_size_risk, dependency_risk, state_penalty)

    def evaluate(self, node: dict) -> dict:
        """
        Full Authority Vector evaluation for one infrastructure node.
        A = C × (1 − Rs) × S × Roll
        """
        C    = self._get_confidence(node.get("CPU_Percent", 100))
        Rs   = self._get_risk(node)
        S    = round(min(1.0, math.log10(max(1, node.get("Monthly_Waste", 0))) / 3.0), 2)
        Roll = 1.0 if node.get("Has_Snapshot") else 0.0
        A    = round(C * (1.0 - Rs) * S * Roll, 4)

        waste     = node.get("Monthly_Waste", 0)
        action    = "MONITOR"
        reason    = "Waste signal below intervention threshold. Passive monitoring active."
        auto_mode = False
        to_where  = "No migration required."

        if Roll == 0.0:
            action   = "BLOCK"
            reason   = (
                f"One-Way Door detected. No snapshot exists for `{node['Instance_ID']}`. "
                f"KARMIN enforces reversibility — irreversible actions are prohibited. "
                f"Resolution: Create an AMI/snapshot and re-queue for next sweep."
            )
            to_where = "Blocked at safety gate. No infrastructure mutation performed."

        elif A > 0.8 and Rs < 0.2:
            action    = "AUTO_TERMINATE"
            auto_mode = True
            reason    = (
                f"Maximum authority granted. Confidence={int(C*100)}%, "
                f"Blast radius={int(Rs*100)}% (fully isolated), "
                f"Authority Vector A={A} exceeds 0.8 threshold. "
                f"Zero human approval required. Immediate capital harvest executed."
            )
            to_where  = (
                f"Instance terminated. EBS volumes detached and flagged for deletion. "
                f"Rollback command: `aws ec2 restore-from-snapshot --id {node['Instance_ID']}`"
            )

        elif A > 0.5:
            action   = "TERMINATE"
            reason   = (
                f"Authority Vector A={A} clears 0.5 threshold. "
                f"Confidence={int(C*100)}%, Blast radius={int(Rs*100)}%. "
                f"Capital recovery authorized. Snapshot exists — Two-Way Door confirmed."
            )
            to_where = (
                f"Instance stopped and scheduled for termination. "
                f"Rollback command: `aws ec2 restore-from-snapshot --id {node['Instance_ID']}`"
            )

        elif Rs > 0.7 and S > 0.5:
            action   = "DOWNSIZE"
            reason   = (
                f"Waste confirmed (S={S}) but blast radius too high for termination (Rs={int(Rs*100)}%). "
                f"Service is stateful or has active RDS sync calls ({node.get('RDS_Sync_Calls_Per_Sec', 0)}/sec). "
                f"Safe downsizing applied — instance type reduced to next smaller tier."
            )
            to_where = (
                f"Instance type reduced. Storage_GB={node.get('Storage_GB', 0)}GB retained. "
                f"Dependencies preserved. No connection disruption."
            )

        return {
            "id":              node["Instance_ID"],
            "service":         node.get("ServiceName", ""),
            "action":          action,
            "auto":            auto_mode,
            "reason":          reason,
            "to_where":        to_where,
            "waste":           waste,
            "arr":             round(waste * 12, 2),
            "C":               C,
            "Rs":              Rs,
            "S":               S,
            "Roll":            Roll,
            "A":               A,
            "confidence_pct":  f"{int(C * 100)}%",
            "rollback":        f"aws ec2 restore-from-snapshot --id {node['Instance_ID']}" if Roll == 1.0 else "N/A",
            "storage_gb":      node.get("Storage_GB", 0),
            "is_stateful":     node.get("Is_Stateful", False),
            "has_snapshot":    node.get("Has_Snapshot", False),
            "rds_sync_calls":  node.get("RDS_Sync_Calls_Per_Sec", 0),
            "z_score":         node.get("ZScore", 0.0),
            "newton_ratio":    node.get("Newton_Ratio", 0.0),
            "tpmad":           node.get("TPMAD_Score", 0.0),
            "vote_count":      node.get("Vote_Count", 0),
        }

    def process_fleet(self, fleet: list) -> list:
        """
        Runs full sovereign sweep across all anomaly-confirmed nodes.
        Tracks monthly recovery and ARR reallocated.
        """
        self.total_monthly_savings = 0.0
        self.total_arr_recovered   = 0.0
        self.actions_executed      = 0
        results = []

        for node in fleet:
            res = self.evaluate(node)
            results.append(res)
            if res["action"] in ["AUTO_TERMINATE", "TERMINATE", "DOWNSIZE"]:
                self.total_monthly_savings += res["waste"]
                self.actions_executed      += 1

        self.total_arr_recovered = round(self.total_monthly_savings * 12, 2)
        self.total_monthly_savings = round(self.total_monthly_savings, 2)
        return results

    def get_summary(self) -> dict:
        return {
            "mode":             self.mode,
            "total_actions":    self.actions_executed,
            "monthly_recovery": self.total_monthly_savings,
            "arr_recovered":    self.total_arr_recovered,
        }

# STEP 5: NARRATOR  
class KarminNarrator:
    def __init__(self):
        self.openers      = ["Infrastructure Audit:", "Financial Leakage Identified:", "ROI Alert:", "Critical Anomaly:"]
        self.impact_verbs = ["hemorrhaging", "wasting", "draining", "bleeding"]
        self.conclusions  = ["Autonomous optimization justified.", "Capital recovery approved.", "Safe remediation path identified."]

    def generate_explanation(self, result: dict) -> str:
        if result.get("status", result.get("action", "")) in ["IGNORED", "MONITOR"]:
            return "Systems nominal. KARMIN detected no anomaly strong enough for autonomous intervention."
        inst    = result.get("id", "Unknown")
        action  = result.get("status", result.get("action", "ACTION"))
        savings = result.get("savings", result.get("waste", 0))
        C       = result.get("C", 0)
        Rs      = result.get("Rs", 0)
        A       = result.get("A", 0)
        opener  = random.choice(self.openers)
        verb    = random.choice(self.impact_verbs)
        end     = random.choice(self.conclusions)
        return (
            f"**{opener}** Node `{inst}` is {verb} approximately **${savings}/mo**. "
            f"KARMIN computed Confidence={int(C*100)}%, Dependency Risk={int(Rs*100)}%, "
            f"Authority Vector A={A}. Autonomous action **{action}** executed. {end}"
        )

# STEP 6: ACTUATOR  
class KarminActuator:
    def execute_bulk(self, instances):
        time.sleep(0.5)
        return [{"id": i["id"], "action": i.get("status", i.get("action", "UNKNOWN"))} for i in instances]

    def revert_state(self, undo_cache):
        time.sleep(0.5)
        return "Infrastructure restored."

# STEP 7: NLP  
def process_nlp(user_text: str, sovereign_log: list, narrator: KarminNarrator, engine_log: list) -> dict:
    text        = user_text.lower()
    instance_id = re.search(r"i-[a-z0-9]{6,20}", text)
    instance_id = instance_id.group(0) if instance_id else None

    if any(w in text for w in ["bill", "cost", "spending", "money", "saved", "arr", "recover"]):
        agent_total  = sum(r["savings"] for r in sovereign_log if r["status"] in ["TERMINATED", "RESIZED"])
        engine_total = sum(r["waste"]   for r in engine_log    if r["action"]  in ["AUTO_TERMINATE", "TERMINATE", "DOWNSIZE"])
        arr          = round(engine_total * 12, 2)
        return {
            "human_summary": (
                f"Autonomous sweep complete. "
                f"Agent recovered **${agent_total:.2f}/mo** | "
                f"Engine recovered **${engine_total:.2f}/mo** | "
                f"ARR reallocated: **${arr:.2f}** across {len(sovereign_log) + len(engine_log)} evaluated instances."
            )
        }

    if instance_id:
        # Search both logs
        item = next((r for r in sovereign_log if r["id"] == instance_id), None)
        if not item:
            item = next((r for r in engine_log if r["id"] == instance_id), None)
        if item:
            return {"human_summary": narrator.generate_explanation(item)}

    return {"human_summary": "Incomplete telemetry request. Provide an Instance ID or ask about the bill."}

# MOCK DATA 
def generate_mock_raw_payload():
    return pd.DataFrame([
        {"ServiceName": "AmazonEC2", "Cost_USD": round(random.uniform(40, 50), 2), "CPU_Percent": random.randint(2, 15),  "Traffic": random.randint(300,  700),  "Actual_Val": 5,  "Provisioned_Cap": 100, "Dependency": False, "Instance_ID": "i-0abc123"},
        {"ServiceName": "AmazonRDS", "Cost_USD": round(random.uniform(18, 22), 2), "CPU_Percent": random.randint(20, 50), "Traffic": random.randint(200,  500),  "Actual_Val": 25, "Provisioned_Cap": 100, "Dependency": False, "Instance_ID": "i-0def456"},
        {"ServiceName": "AmazonS3",  "Cost_USD": round(random.uniform(11, 13), 2), "CPU_Percent": random.randint(60, 90), "Traffic": random.randint(800, 1500),  "Actual_Val": 85, "Provisioned_Cap": 100, "Dependency": True,  "Instance_ID": "i-0ghi789"},
    ])

def generate_mock_history(current_profile_df):
    records = []
    for _, row in current_profile_df.iterrows():
        for _ in range(4):
            baseline_waste = max(0.5, row["Potential_Waste_USD"] * random.uniform(0.15, 0.4))
            records.append({
                "ServiceName":         row["ServiceName"],
                "Instance_ID":         row["Instance_ID"],
                "Cost_USD":            row["Cost_USD"] * random.uniform(0.9, 1.0),
                "CPU_Percent":         row["CPU_Percent"] * random.uniform(1.0, 1.2),
                "Traffic":             row["Traffic"] * random.uniform(0.8, 1.1),
                "Potential_Waste_USD": baseline_waste,
                "UER":                 row["UER"] * random.uniform(1.2, 1.6),
                "Dependency":          row["Dependency"]
            })
        records.append(row.to_dict())
    return pd.DataFrame(records)

# APP EXECUTION
st.title("☁️ KARMIN Autonomous Cloud Cost Intelligence")
st.caption("Fully autonomous FinOps agent · Ensemble Detection → Sovereign Engine → Action Summary")

strategy = st.sidebar.radio("Optimization Strategy", ["COST", "GREEN"])
engine_mode = st.sidebar.selectbox("Engine Mode", ["ESG_COMPLIANCE", "COST_AGGRESSIVE", "CONSERVATIVE"])

if st.sidebar.button("🔄 Re-run Full Sweep"):
    st.session_state.sweep_done              = False
    st.session_state.sovereign_log           = []
    st.session_state.total_saved             = 0.0
    st.session_state.engine_log              = []
    st.session_state.engine_done             = False
    st.session_state.engine_monthly_recovery = 0.0
    st.session_state.engine_arr              = 0.0
    st.rerun()

# Steps 1 + 2
raw_payload   = generate_mock_raw_payload()
ingestor      = KarminUniversalServiceIngestor(provider_name="AWS", linked_account_id="123456789012")
normalized_df = ingestor.normalize_service_matrix(raw_payload)
profiler      = KarminContextProfiler()
profile_df    = profiler.profile_batch(normalized_df)

# Step 3: Ensemble Anomaly Engine
history_df = generate_mock_history(profile_df)
ensemble   = KarminEnsembleEngine()
results    = []
for svc in history_df["ServiceName"].unique():
    out = ensemble.evaluate_service(history_df[history_df["ServiceName"] == svc])
    if out:
        results.append(out)
ensemble_df = pd.DataFrame(results)

# Step 4/5: Sovereign Agent sweep — runs once, cached
narrator  = KarminNarrator()
actuator  = KarminActuator()
agent     = KarminSovereignAgent()

if not st.session_state.sweep_done:
    with st.spinner("⚡ KARMIN Sovereign Agent Sweep in progress..."):
        log = agent.run_sweep(ensemble_df)
        actuator.execute_bulk(log)
        st.session_state.sovereign_log = log
        st.session_state.total_saved   = agent.total_saved_this_pass
        st.session_state.sweep_done    = True

sovereign_log = st.session_state.sovereign_log

# Step 5: Sovereign Engine — bridge → enrich → evaluate fleet
sovereign_engine = KarminSovereignEngine(mode=engine_mode)

if not st.session_state.engine_done:
    with st.spinner("🧠 KARMIN Sovereign Engine evaluating fleet..."):
        # Bridge: enrich ALL anomaly rows from ensemble with infrastructure metadata
        enriched_fleet = []
        for _, row in ensemble_df.iterrows():
            if row.get("Anomaly", False):
                enriched_node = bridge_from_brain(row.to_dict())
                enriched_fleet.append(enriched_node)

        engine_results = sovereign_engine.process_fleet(enriched_fleet)
        summary        = sovereign_engine.get_summary()

        st.session_state.engine_log              = engine_results
        st.session_state.engine_monthly_recovery = summary["monthly_recovery"]
        st.session_state.engine_arr              = summary["arr_recovered"]
        st.session_state.engine_done             = True

engine_log = st.session_state.engine_log

# TOP DASHBOARD
terminated     = [r for r in sovereign_log if r["status"] == "TERMINATED"]
resized        = [r for r in sovereign_log if r["status"] == "RESIZED"]
blocked_agent  = [r for r in sovereign_log if r["status"] == "BLOCKED"]
auto_term      = [r for r in engine_log    if r["action"] == "AUTO_TERMINATE"]
term_engine    = [r for r in engine_log    if r["action"] == "TERMINATE"]
downsize       = [r for r in engine_log    if r["action"] == "DOWNSIZE"]
blocked_engine = [r for r in engine_log    if r["action"] == "BLOCK"]

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("💰 Agent Monthly Recovery",   f"${st.session_state.total_saved:.2f}")
col2.metric("📈 Engine Monthly Recovery",  f"${st.session_state.engine_monthly_recovery:.2f}")
col3.metric("📊 ARR Reallocated",          f"${st.session_state.engine_arr:.2f}")
col4.metric("⚡ Auto-Terminated",          len(auto_term))
col5.metric("🚫 Blocked (Safety Gate)",    len(blocked_agent) + len(blocked_engine))

st.divider()

# INCOMING CLOUD METRICS 
st.subheader("📊 Incoming Cloud Metrics")
st.dataframe(profile_df[[
    "ServiceName", "Cost_USD", "CPU_Percent", "Traffic",
    "Utilization_Pct", "Potential_Waste_USD", "UER", "Status"
]], use_container_width=True)


# SOVEREIGN AGENT LOG  
st.subheader("🤖 KARMIN Sovereign Agent Log")
st.caption("Step 4B output — Agent-level decisions before infrastructure enrichment.")

for r in sovereign_log:
    status  = r["status"]
    r_id    = r["id"]
    savings = r["savings"]
    reason  = r["reason"]
    A       = r["A"]
    C       = r["C"]
    Rs      = r["Rs"]
    votes   = r["vote_count"]

    if status == "TERMINATED":
        st.success(
            f"✅ **TERMINATED** · `{r_id}` ({r['service']})\n\n"
            f"**Reason:** {reason}\n\n"
            f"**Monthly Savings:** ${savings:.2f} | **A:** {A} | "
            f"**Consensus:** {votes}/3 | **Confidence:** {int(C*100)}% | **Risk:** {int(Rs*100)}%"
        )
    elif status == "RESIZED":
        st.info(
            f"🔁 **RESIZED** · `{r_id}` ({r['service']})\n\n"
            f"**Reason:** {reason}\n\n"
            f"**Monthly Savings:** ${savings:.2f} | **A:** {A} | "
            f"**Consensus:** {votes}/3 | **Confidence:** {int(C*100)}% | **Risk:** {int(Rs*100)}%"
        )
    elif status == "BLOCKED":
        st.error(
            f"🚫 **BLOCKED** · `{r_id}` ({r['service']})\n\n"
            f"**Reason:** {reason}\n\n"
            f"**Unrecovered Waste:** ${savings:.2f}/mo | **Fix:** Create snapshot → re-queue."
        )
    else:
        st.warning(f"👁️ **MONITORING** · `{r_id}` ({r['service']})\n\n**Reason:** {reason}")

st.divider()


# STEP 6: SOVEREIGN ENGINE ACTION SUMMARY
st.subheader("⚙️ KARMIN Sovereign Engine — Action Summary")
st.caption(
    f"Step 4C output · Mode: **{engine_mode}** · "
    f"Authority Vector: A = Confidence × (1 − Risk) × Savings × Safety"
)

#Engine Summary Banner
summary_cols = st.columns(4)
summary_cols[0].metric("✅ Total Actions Executed",  len([r for r in engine_log if r["action"] != "MONITOR"]))
summary_cols[1].metric("⚡ AUTO_TERMINATE",          len(auto_term))
summary_cols[2].metric("🔽 DOWNSIZE",                len(downsize))
summary_cols[3].metric("🚫 BLOCK",                   len(blocked_engine))

st.markdown("---")

#Per-Node Action Cards
for r in engine_log:
    action     = r["action"]
    r_id       = r["id"]
    service    = r["service"]
    waste      = r["waste"]
    arr        = r["arr"]
    reason     = r["reason"]
    to_where   = r["to_where"]
    A          = r["A"]
    C          = r["C"]
    Rs         = r["Rs"]
    Roll       = r["Roll"]
    confidence = r["confidence_pct"]
    rollback   = r["rollback"]
    auto_flag  = r["auto"]

    #AUTO_TERMINATION
    if action == "AUTO_TERMINATE":
        st.success(
            f"⚡ **AUTO_TERMINATE** {'[FULLY AUTONOMOUS]' if auto_flag else ''} · "
            f"`{r_id}` ({service})\n\n"
            f"**WHAT:** Instance automatically terminated without human approval.\n\n"
            f"**WHY:** {reason}\n\n"
            f"**TO WHERE:** {to_where}\n\n"
            f"💰 **Monthly Recovery:** ${waste:.2f} | 📈 **ARR:** ${arr:.2f} | "
            f"🔢 **A={A}** | 🧠 **Confidence={confidence}** | "
            f"⚖️ **Blast Radius={int(Rs*100)}%** | 🛡️ **Rollback Ready**\n\n"
            f"↩️ **Undo:** `{rollback}`"
        )

    #TERMINATE
    elif action == "TERMINATE":
        st.success(
            f"✅ **TERMINATE** · `{r_id}` ({service})\n\n"
            f"**WHAT:** Instance terminated. Snapshot confirmed before execution.\n\n"
            f"**WHY:** {reason}\n\n"
            f"**TO WHERE:** {to_where}\n\n"
            f"💰 **Monthly Recovery:** ${waste:.2f} | 📈 **ARR:** ${arr:.2f} | "
            f"🔢 **A={A}** | 🧠 **Confidence={confidence}** | "
            f"⚖️ **Blast Radius={int(Rs*100)}%** | 🛡️ **Snapshot Exists**\n\n"
            f"↩️ **Undo:** `{rollback}`"
        )

    # --- DOWNSIZE ---
    elif action == "DOWNSIZE":
        st.info(
            f"🔽 **DOWNSIZE** · `{r_id}` ({service})\n\n"
            f"**WHAT:** Instance type reduced to next smaller tier. No termination.\n\n"
            f"**WHY:** {reason}\n\n"
            f"**TO WHERE:** {to_where}\n\n"
            f"💰 **Monthly Recovery:** ${waste:.2f} | 📈 **ARR:** ${arr:.2f} | "
            f"🔢 **A={A}** | 🧠 **Confidence={confidence}** | "
            f"⚖️ **Blast Radius={int(Rs*100)}%** (too high for termination)\n\n"
            f"↩️ **Undo:** `{rollback}`"
        )

    #BLOCK
    elif action == "BLOCK":
        st.error(
            f"🚫 **BLOCK** · `{r_id}` ({service})\n\n"
            f"**WHAT:** Action refused. Safety override engaged.\n\n"
            f"**WHY:** {reason}\n\n"
            f"**TO WHERE:** {to_where}\n\n"
            f"⚠️ **Unrecovered Waste:** ${waste:.2f}/mo | "
            f"🔢 **A={A}** (Safety gate: Roll=0.0)\n\n"
            f"🔧 **Fix:** Create AMI/snapshot → KARMIN will auto-evaluate on next sweep."
        )

    #MONITOR
    else:
        st.warning(
            f"👁️ **MONITOR** · `{r_id}` ({service})\n\n"
            f"**WHAT:** No action taken. Passive monitoring continues.\n\n"
            f"**WHY:** {reason}\n\n"
            f"🔢 **A={A}** | 🧠 **Confidence={confidence}** | ⚖️ **Risk={int(Rs*100)}%**"
        )

st.divider()

# EFFICIENCY PANEL AND TAGS
st.subheader("💡 Efficiency Analysis")
for _, row in profile_df.iterrows():
    efficiency = row["Traffic"] / row["Cost_USD"] if row["Cost_USD"] > 0 else 0
    st.write(f"**{row['ServiceName']}** → Efficiency Score: `{efficiency:.2f}`")
    if efficiency < 15:
        st.warning(f"{row['ServiceName']}: Low efficiency → Reduce / Resize")
    else:
        st.success(f"{row['ServiceName']}: High efficiency → Protect")


tab1, tab2, tab3, tab4 = st.tabs(["🛡️ Agent Triage", "⚙️ Engine Audit", "🧠 Ensemble Engine", "💻 NLP Terminal"])
with tab1:
    st.caption("Step 4B — Agent-level detailed audit cards.")
    for r in sovereign_log:
        label = f"{r['status']} · {r['id']} · ${r['savings']:.2f}/mo"
        with st.expander(label):
            st.markdown(f"**Service:** {r['service']}")
            st.markdown(f"**Status:** `{r['status']}`")
            st.markdown(f"**Reason:** {r['reason']}")
            st.json({
                "Authority_Vector_A": r["A"],
                "Confidence_C":       r["C"],
                "Dependency_Risk_Rs": r["Rs"],
                "Savings_Impact_S":   r["S"],
                "Rollback_Readiness": r["Roll"],
                "Detector_Votes":     f"{r['vote_count']}/3",
                "ZScore":             r["z_score"],
                "Newton_Ratio":       r["newton_ratio"],
                "TPMAD":              r["tpmad"],
                "CPU_Percent":        r["cpu"],
                "UER":                r["uer"],
            })

with tab2:
    st.caption("Step 4C — Sovereign Engine full audit trail with infrastructure enrichment.")
    for r in engine_log:
        label = f"{r['action']} · {r['id']} ({r['service']}) · ${r['waste']:.2f}/mo | ARR ${r['arr']:.2f}"
        with st.expander(label):
            st.markdown(f"**Action:** `{r['action']}` {'⚡ AUTONOMOUS' if r['auto'] else ''}")
            st.markdown(f"**What was done:** {r['reason']}")
            st.markdown(f"**Where/How:** {r['to_where']}")
            st.json({
                "Authority_Vector_A":       r["A"],
                "Confidence_C":             r["C"],
                "Blast_Radius_Rs":          r["Rs"],
                "Savings_Impact_S":         r["S"],
                "Safety_Roll":              r["Roll"],
                "Monthly_Waste_USD":        r["waste"],
                "ARR_USD":                  r["arr"],
                "Storage_GB":               r["storage_gb"],
                "Is_Stateful":              r["is_stateful"],
                "Has_Snapshot":             r["has_snapshot"],
                "RDS_Sync_Calls_Per_Sec":   r["rds_sync_calls"],
                "Ensemble_ZScore":          r["z_score"],
                "Ensemble_Newton_Ratio":    r["newton_ratio"],
                "Ensemble_TPMAD":           r["tpmad"],
                "Ensemble_Vote_Count":      f"{r['vote_count']}/3",
                "Rollback_Command":         r["rollback"],
            })

with tab3:
    st.dataframe(ensemble_df, use_container_width=True)

with tab4:
    query = st.chat_input("Command Karmin (e.g., 'Check i-0abc123' or 'how much did we save?' or 'show ARR')")
    if query:
        st.chat_message("user").write(query)
        res = process_nlp(query, sovereign_log, narrator, engine_log)
        st.chat_message("assistant").write(res["human_summary"])
