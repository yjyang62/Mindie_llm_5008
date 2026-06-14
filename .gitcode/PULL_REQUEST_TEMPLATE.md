<!--
PR描述模板更新日期：20251225
-->

# 合入背景
> 请描述为什么要做这个PR内的改动。\
> 如涉及，请关联前序PR或同特性/需求下的其他PR。\
> 如果是修复之前PR引入的问题，请关联引入问题的PR。\
> 注意：`Fixes #ISSUE ID`会自动关闭issue，如问题部分解决请不要使用`Fixes`，可以用`Fix part of #ISSUE ID`替代.

# 修改内容
> 请描述修改内容的具体实现，涉及哪些组件之间进行交互，可以用1、2、3、...进行罗列。\
> 如果是需求或者重构类的PR，需要补充详细设计文档（说明上下游组件关系、时序图、类图、DFX能力等内容）。

# 资料变更
> 请确认是否涉及资料变更。如涉及，需要在PR中体现，并简要说明修改内容。如不涉及，需填写“不涉及”。

# 接口变更
> 请确认是否涉及跨代码仓或者客户面可见的接口变更。如涉及，需要详细说明接口以及对应的变更内容，同时需要在资料中体现。如不涉及，需填写“不涉及”。

# 测试结果
> 请说明测试场景，测试方法以及测试结果。\
> 测试用例设计时需考虑硬件、部署方式、功能、性能、精度、显存等维度。

# CheckList
> PR提交人对以下CheckList自检项进行全量自检，自检通过或不涉及，均修改 [ ] 为 [x]。

- [ ] 代码注释完备
- [ ] 正确记录错误日志
- [ ] 进行了返回值校验 (禁止使用void屏蔽安全函数、自研函数返回值；考虑接口的异常场景；调用底层组件接口时，需要进行返回值校验)
- [ ] 进行了空指针校验
- [ ] 若存在资源申请，使用后资源被正确的释放了
- [ ] 若涉及多线程场景，考虑了并发场景，不存在死锁问题
- [ ] 按照[代码仓中提供的格式模板](https://gitcode.com/Ascend/MindIE-LLM/blob/master/.clang-format)，使用clang-format工具格式化代码
- [ ] 符合Ascend社区的编码规范。[C++ 语言编程指导](https://gitcode.com/Ascend/community/blob/master/docs/contributor/Ascend-cpp-coding-style-guide.md) | [C++ 语言安全编程指导](https://gitcode.com/Ascend/community/blob/master/docs/contributor/Ascend-cpp-secure-coding-guide.md)
