# eval_results GitHub Upload Note 508

本目录下的非 checkpoint 评测产物已纳入 Git 上传范围，包括：

- `test_results*` 下的 `test_all_preds.npy` / `test_all_gts.npy`
- `duration_summary.json`
- `duration_per_chunk.csv`
- `gt_target_completion_aggregate.csv`
- `vis*` / `vis_trajectory*` 下的 PNG 可视化图

未直接上传 checkpoint 文件：

- `eval_results/**/checkpoints/**`

原因：

- 当前 `eval_results` 总体约 80GB。
- checkpoint 内存在大量 100MB 以上文件，最大单文件约 3GB。
- GitHub 普通 Git 推送会拒绝超过 100MB 的文件。
- 即使使用 Git LFS，也需要额外确认 GitHub LFS 配额和带宽，不能默认把 80GB checkpoint 推到远端。

如需长期保存 checkpoint，建议使用 OSS / HuggingFace Hub / GitHub Release + LFS 配额确认后的专用流程。

