// 获取用户主目录路径
let userHome = Components.classes["@mozilla.org/file/directory_service;1"]
    .getService(Components.interfaces.nsIProperties)
    .get("Home", Components.interfaces.nsIFile).path;

pref("extensions.text_exporter.baseDirTitleAbstract", userHome + "\\Zotero-export\\titleAbstract");
pref("extensions.text_exporter.baseDirFullText", userHome + "\\Zotero-export\\fulltext");
pref("extensions.text_exporter.contentTypeFilter_html_checkbox", false);
pref("extensions.text_exporter.contentTypeFilter_pdf_checkbox", false);
