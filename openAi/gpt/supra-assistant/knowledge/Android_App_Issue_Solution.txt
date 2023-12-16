关于一种解决android app损坏，无法卸载和覆盖安装问题的解决方法：

问题描述：
android手机安装apk时报错，提示系统存在签名不同的app（具备launcher intent filter），导致安装失败，尝试卸载apk，但是卸载失败，但是在launcher中无法找到该app，应该如何解决？

安装报错信息：
Unable to install /path/to/app.apk
com.android.ddmlib.InstallException: INSTALL_FAILED_UPDATE_INCOMPATIBLE: Package com.example.app signatures do not match previously installed version; ignoring!

卸载报错信息：
Failure [DELETE_FAILED_INTERNAL_ERROR]

解决方案，可以采用如下步骤解决这个问题：
1. 首先使用这个命令得到app的apk安装包目录地址：`adb shell pm path com.example.app`
2. 然后把这个地址下的apk安装包拷贝到电脑上：`adb pull /on/device/path/to/app.apk`
3. 然后重新安装这个apk包，修复系统上的app：`adb install /path/to/app.apk`
4. 最后卸载这个版本的app：`adb uninstall com.example.app`

这几步操作之后就可以安装成功新版本的apk了
