    // Start of Selection
    // import FilePicker from 'zotero/modules/filePicker';
    // import FilePicker from 'zotero/modules/filePicker';
    Components.utils.import("resource://gre/modules/Services.jsm");
    
    Zotero.TextExporter = {
    
        log(msg) {
            Zotero.debug("Title Abstract Export: " + msg);
        },
    
        sanitizeFileName(fileName) {
            // 这里的 max length 要保证：
            // 1）加上 dir path 后不超过 255 个字符
            // 2）加上 .txt 扩展名后，小于绘图代码中的 file_names.append(os.path.basename(path)[:150])，
            // 否则在那里截断将导致文件名不匹配
            let sanitized = fileName.replace(/[<>:"/\\|?*]/g, '_').substring(0, 145);
            
            // 记录已处理的文件名
            if (!this.processedFileNames) {
                this.processedFileNames = new Set();
            }
            
            // 检查文件名是否已存在，如果存在则添加序号
            let uniqueSanitized = sanitized;
            let counter = 1;
            while (this.processedFileNames.has(uniqueSanitized)) {
                uniqueSanitized = sanitized.substring(0, 140) + '_' + counter;
                counter++;
            }
            
            // 将新的唯一文件名添加到已处理集合中
            this.processedFileNames.add(uniqueSanitized);
            
            return uniqueSanitized;
        },
    
    
        getPythonPath() {
            let envService = Components.classes["@mozilla.org/process/environment;1"]
                             .getService(Components.interfaces.nsIEnvironment);
            let pythonPath = envService.get("PYTHON_PATH") || envService.get("PATH");
            
            if (pythonPath) {
                let paths = pythonPath.split(';');
                for (let path of paths) {
                    let file = Components.classes["@mozilla.org/file/local;1"]
                               .createInstance(Components.interfaces.nsIFile);
                    file.initWithPath(path);
                    file.append("python.exe");
                    if (file.exists()) {
                        return file.path;
                    }
                }
            }
            
            // 如果在环境变量中找不到，返回默认路径
            return this.getUserHomeDirectory() + '\\AppData\\Local\\Programs\\Python\\Python312\\python.exe';
        },
    

    
        runPythonScript(scriptPath, scriptArgs = []) {
            try {
                let pythonPath = this.getPythonPath();

                setTimeout(() => {
                    Zotero.Utilities.Internal.exec(pythonPath, [scriptPath, ...scriptArgs]);
                    this.log('Python脚本已开始执行');
                }, 0);

                this.showPopup('Python脚本', 'Python脚本已开始运行');
            } catch (e) {
                Zotero.logError(e);
                this.showPopup('错误', '启动Python脚本时发生错误: ' + e.message, true);
            }
        },
    
        
        async buildOmniObject() {
            var items = ZoteroPane.getSelectedItems();
            if (!items.length) {
                this.showPopup('错误', '请先选择要处理的条目', true);
                return null;
            }
    
            let obj = {
                data: []
            };
    
            for (let item of items) {
                if (item.isRegularItem()) {
                    let itemData = {
                        id: item.id,
                        title: item.getField('title'),
                        year: item.getField('year'),
                        authors: item.getCreators().map(creator => `${creator.firstName} ${creator.lastName}`).join(', '),
                        abstract: item.getField('abstractNote'),
                        fulltext: []
                    };
    
                    let attachments = item.getAttachments();
                    for (let attachmentID of attachments) {
                        let attachment = Zotero.Items.get(attachmentID);
                        let htmlChecked = Zotero.Prefs.get('extensions.text_exporter.contentTypeFilter_html_checkbox', true);
                        let pdfChecked = Zotero.Prefs.get('extensions.text_exporter.contentTypeFilter_pdf_checkbox', true);
                        Zotero.debug("HTML选项状态: " + htmlChecked);
                        Zotero.debug("PDF选项状态: " + pdfChecked);
                        Zotero.debug("附件内容类型: " + attachment.attachmentContentType);
                        
                        if ((htmlChecked && attachment.attachmentContentType === 'text/html') ||
                            (pdfChecked && attachment.attachmentContentType === 'application/pdf')) {
                            try {
                                let text = await attachment.attachmentText;
                                if (text) {
                                    itemData.fulltext.push(text);
                                }
                            } catch (e) {
                                this.log('无法提取 "' + item.getField('title') + '" 的附件全文: ' + e);
                            }
                        }
                    }
                    obj.data.push(itemData);
                }
            }
    
            // Zotero.debug("导出对象内容:");
            // Zotero.debug(JSON.stringify(obj, null, 2));
    
            return obj;
        },
    
        async exportItemsAndRunPython() {
            await this.exportItems();
            let obj = Zotero.TextExporter.omniObject; // 使用已保存的对象
            if (!obj) {
                this.showPopup('错误', '获取数据失败，无法进行聚类', true);
                return;
            }
            obj.option = {
                embedding_input: 'abstract',
                embedding_model: 'local'
            };
            // 将obj转换为JSON字符串并复制到剪贴板
            let jsonString = JSON.stringify(obj);
            this.copyToClipboard(jsonString);
            this.runPythonScript('D:\\My_Codes\\document_clustering_GUI\\main.py');
        },
        
        async exportItems() {
            let obj = await this.buildOmniObject();
            if (!obj) return;
    
            Zotero.TextExporter.omniObject = obj;
    
            var userHome = this.getUserHomeDirectory();
            var baseDir = Zotero.Prefs.get('extensions.text_exporter.baseDirTitleAbstract', true) || (userHome + '\\Zotero-export\\titleAbstract');
            this.log('baseDirTitleAbstract: ' + baseDir);
            let emptyAbstractCount = 0;
    
            for (let itemData of obj.data) {
                let fileName = this.sanitizeFileName(`${itemData.year} - ${itemData.title} _ { ${itemData.authors} }`) + '.txt';
                let filePath = baseDir + '\\' + fileName;
                
                let data = `${itemData.year} - ${itemData.title}\n\n${itemData.authors}\n\n`;
                if (itemData.abstract) {
                    data += itemData.abstract + "\n\n";
                } else {
                    emptyAbstractCount++;
                }
    
                try {
                    await Zotero.File.putContentsAsync(filePath, data);
                    itemData.exportedPath = filePath;
                } catch (e) {
                    Zotero.logError(e);
                    this.showPopup('错误', '导出 "' + itemData.title + '" 时发生错误: ' + e, true);
                }
            }
    
            this.showPopup('导出完成', `空摘要数量：${emptyAbstractCount}/${obj.data.length}`);
        },


        // 重构后的函数：合并项目信息并导出
        async exportItemsConcatAndCopyCore(includeAbstract = true, isFullText = false) {
            let obj = await this.buildOmniObject();
            if (!obj) {
                this.showPopup('错误', '获取数据失败，无法进行导出', true);
                return;
            }

            // 合并项目信息
            let concatenatedItems = obj.data
                .map(item => {
                    let itemContent = `${item.year} - ${item.title} _ { ${item.authors} }\n\n`;
                    if (isFullText) {
                        if (item.fulltext && item.fulltext.length > 0) {
                            itemContent += item.fulltext.join('\n\n');
                        }
                    } else {
                        if (includeAbstract && item.abstract) {
                            itemContent += item.abstract + "\n\n";
                        }
                    }
                    return itemContent;
                })
                .filter(content => content.trim() !== '')
                .join('\n\n');

            if (!concatenatedItems) {
                this.showPopup('错误', '没有可用的内容进行导出', true);
                return;
            }

            // 生成文件名和路径
            let fileName, baseDir;
            let userHome = Components.classes["@mozilla.org/file/directory_service;1"]
                .getService(Components.interfaces.nsIProperties)
                .get("Home", Components.interfaces.nsIFile).path;
            if (isFullText) {
                fileName = this.sanitizeFileName('merged_fulltext') + '.txt';
                baseDir = Zotero.Prefs.get('extensions.text_exporter.baseDirFullText', true) || (userHome + '\\Zotero-export\\fulltext');
            } else {
                fileName = this.sanitizeFileName(includeAbstract ? 'merged_items' : 'merged_basic_info') + '.txt';
                baseDir = Zotero.Prefs.get('extensions.text_exporter.baseDirTitleAbstract', true) || (userHome + '\\Zotero-export\\titleAbstract');
            }
            let filePath = baseDir + '\\' + fileName;

            try {
                // 写入文件
                await Zotero.File.putContentsAsync(filePath, concatenatedItems);
                // 复制到剪贴板
                this.copyToClipboard(concatenatedItems);
                // 运行Python脚本
                let scriptPath = 'D:\\My_Codes\\create-file-from-clipboard-text\\main.py';
                let exportDir = isFullText ? 'D:\\My_Working_Spaces\\Zotero-export\\fulltext' : 'D:\\My_Working_Spaces\\Zotero-export\\titleAbstract';
                this.runPythonScript(scriptPath, [fileName, exportDir]);
                // 显示成功弹窗
                let message;
                if (isFullText) {
                    message = '合并的全文（包含标题、作者和年份）已导出并复制到剪贴板';
                } else {
                    message = includeAbstract ? '合并的项目信息（包含摘要）已导出并复制到剪贴板' : '合并的基本项目信息已导出并复制到剪贴板';
                }
                this.showPopup('导出完成', message);
            } catch (e) {
                Zotero.logError(e);
                this.showPopup('错误', '导出合并项目信息时发生错误: ' + e, true);
            }
        },

        // 导出包含摘要的合并项目信息
        async exportItemsConcatAndCopy() {
            await this.exportItemsConcatAndCopyCore(true, false);
        },

        // 导出不包含摘要的合并项目基本信息
        async exportItemsBasicInfoConcatAndCopy() {
            await this.exportItemsConcatAndCopyCore(false, false);
        },

        // 导出合并全文
        async exportFullTextConcatAndCopy() {
            await this.exportItemsConcatAndCopyCore(false, true);
        },

    
        async exportFullTextAndRunPython() {
            await this.exportFullText();
            let obj = Zotero.TextExporter.omniObject; // 使用已保存的对象
            if (!obj) {
                this.showPopup('错误', '获取数据失败，无法进行聚类', true);
                return;
            }
            obj.option = {
                embedding_input: 'fulltext',
                embedding_model: 'local'
            };        
            // 将obj转换为JSON字符串并复制到剪贴板
            let jsonString = JSON.stringify(obj);
            this.copyToClipboard(jsonString);
            this.runPythonScript('D:\\My_Codes\\document_clustering_GUI\\main.py');
        },
    
        async exportFullText() {
            let obj = await this.buildOmniObject();
            if (!obj) return;
    
            // 将buildOmniObject()的结果保存为Zotero.TextExporter的一个变量
            Zotero.TextExporter.omniObject = obj;
    
            let userHome = Components.classes["@mozilla.org/file/directory_service;1"]
                .getService(Components.interfaces.nsIProperties)
                .get("Home", Components.interfaces.nsIFile).path;
            var baseDir = Zotero.Prefs.get('extensions.text_exporter.baseDirFullText', true) || (userHome + '\\Zotero-export\\fulltext');
            
            for (let itemData of obj.data) {
                if (itemData.fulltext.length > 0) {
                    let fileName = this.sanitizeFileName(`${itemData.year} - ${itemData.title} _ { ${itemData.authors} }`) + '.txt';
                    let filePath = baseDir + '\\' + fileName;
                    try {
                        Zotero.debug("正在导出文件: " + filePath);
                        let fullTextWithHeader = `${itemData.year} - ${itemData.title} _ { ${itemData.authors} }\n\n${itemData.fulltext.join('\n\n')}`;
                        Zotero.debug("全文内容长度: " + fullTextWithHeader.length + " 字符");
                        await Zotero.File.putContentsAsync(filePath, fullTextWithHeader);
                        // exportedPaths.push(filePath);
                        itemData.exportedPath = filePath; // 将路径写回到itemData中
                    } catch (e) {
                        Zotero.logError(e);
                        this.showPopup('错误', '导出 "' + itemData.title + '" 时发生错误', true);
                    }
                }
            }
    
            this.showPopup('导出完成', '');
        },
    

    
        copyToClipboard(text) {
            var clipboard = Components.classes["@mozilla.org/widget/clipboardhelper;1"].getService(Components.interfaces.nsIClipboardHelper);
            clipboard.copyString(text);
        },
    
        showPopup(title, body, isError = false, timeout = 5) {
            const seconds = 1000;
            const pw = new Zotero.ProgressWindow();
            if (isError) {
                pw.changeHeadline("错误", "chrome://zotero/skin/cross.png", `文本导出器: ${title}`);
            } else {
                pw.changeHeadline(`文本导出器: ${title}`);
            }
            pw.addDescription(body);
            pw.show();
            pw.startCloseTimer(timeout * seconds);
        },
    
        getUserHomeDirectory() {
            let file = Components.classes["@mozilla.org/file/directory_service;1"]
                .getService(Components.interfaces.nsIProperties)
                .get("Home", Components.interfaces.nsIFile);
            return file.path;
        },
    
        initializeDefaultDirectories() {
            let userHome = this.getUserHomeDirectory();
            let exportDirs = [
                userHome + '\\Zotero-export\\titleAbstract',
                userHome + '\\Zotero-export\\fulltext'
            ];
    
            for (let dir of exportDirs) {
                let file = Components.classes["@mozilla.org/file/local;1"]
                    .createInstance(Components.interfaces.nsIFile);
                file.initWithPath(dir);
                if (!file.exists()) {
                    file.create(Components.interfaces.nsIFile.DIRECTORY_TYPE, 0o755);
                    this.log('创建目录: ' + dir);
                }
            }
        },
    
        initialize() {
            this.initializeDefaultDirectories();
            // 其他初始化代码...
        }
    };
    
    // 其他注释掉的函数保持不变
    
    window.addEventListener('load', () => {
        Zotero.TextExporter.initialize();
    }, false);

