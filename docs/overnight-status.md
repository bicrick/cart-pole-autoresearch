# Triple overnight status — 2026-09-20 ~03:45 CT

- VM `cartpole-train-od` RUNNING us-east1-b L4 ~99%/15.6GB; TB http://34.148.138.48:6006/ HTTP 200; uptime ~2d
- **Roles:** A=S1 (restarted LR=1e-4 after NaN); B=H1 pid **145275**; C=H1-var pid **145280**. Markers in `policies/.triple-*`. TRACK_WALLS=1 all.
- **Meters (~u220–230, pre-A-kill):**
  - A S1: hang_align/UUU **-0.98→-0.06** then **NaN blowup @u228** (policy_loss~1e29, full ckpt NaN @u230, eval reward −636). **Killed + quarantined**; cold restart **LR=1e-4** + NaN-guard on continues.
  - B H1: nt_align/UUU →**0.167** nt_UUU →**0.051** reward~187 ent **−0.14** (dead). ENT=0.035+VEL_COST staged; **no mid-kill**; stretch ETA ~05:00 CT.
  - C H1-var: nt_align/UUU →**0.153** nt_UUU ~**0.045** ent **~1.59** — still best H1 signal; leave alone.
- **This fire GO:** mid-kill A only (NaN, not climbing). Quarantine poison ckpt. Harden `quarantine-nan-ckpt.sh` on A/B/C continues; A LR 3e-4→1e-4. B/C undisturbed.
- NEED_USER_PING: **yes** (standing think-out-loud every fire)
