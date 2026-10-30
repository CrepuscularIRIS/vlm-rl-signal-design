# 数据来源与导出边界

- 上游公开实验仓库固定commit：`2c5155d7e629e6eadb043bb8c751035f9132fca9`，见[固定树](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/tree/2c5155d7e629e6eadb043bb8c751035f9132fca9)。
- IMPORTED_FILES.json记录主要拷贝文件的来源标签、源SHA、目标SHA和变换。目标文件发布前如修正文档链接，最终以RELEASE_FILES.json为准。
- CODE_HASHES.json的5个v53核心源码与正式启动SHA一致。没有私有路径改写这些代码。
- PRIVATE_SOURCE_IDENTITIES.json仅保留本地Conversation/训练计数等来源的标签与SHA，不提供完整私聊或机器状态。
- RELEASE_FILES.json记录最终发布材料的文件哈希（自身除外，也排除发布校验回执以免循环）。
- 论文SOURCE笔记保留原核查状态；未重新做13篇精读。原PDF不重发。
- 轨迹沿用上游脱敏：不含完整题面、图像、主机信息；模型复述的已知长题干与伪造图片链接已处理。病例页使用这些公开输出和已有审查记录，不导出新的私有图像。
- 假设文件是用户授权保存的讨论上下文，不是已验证结果。当前没有新增训练/推理/Pro咨询。
