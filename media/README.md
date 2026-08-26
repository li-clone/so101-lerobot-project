# 公开媒体处理规则

当前目录不包含原始视频。发布前从本地评测数据中复制代表性片段到临时目录，完成复核后再上传 GitHub Release；仓库只保存缩略图和索引。

建议片段：v1成功、v1远距离失败、v2近/中/远成功、v2近距离偏抓、v2极远位置重复尝试。

第三视角必须裁掉人物、家庭环境、窗户和无关区域；所有视频移除音轨与元数据。示例命令需要按实际隐私区域调整裁剪参数：

```bash
ffmpeg -i input.mp4 \
  -vf "crop=<width>:<height>:<x>:<y>,scale=960:-2" \
  -an -map_metadata -1 -c:v libx264 -preset slow -crf 26 output.public.mp4
```

逐帧检查开头、结尾、反光和画面边缘。处理完成前不要使用 `git add -f` 绕过媒体忽略规则。
