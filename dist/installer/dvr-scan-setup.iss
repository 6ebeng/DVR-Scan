; DVR-Scan Inno Setup Script
; This file is used by the release workflow to build the Windows installer

#define MyAppName "DVR-Scan"
#define MyAppPublisher "DVR-Scan"
#define MyAppURL "https://github.com/6ebeng/DVR-Scan"

[Setup]
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}/issues
DefaultDirName={autopf}\DVR-Scan
DefaultGroupName=DVR-Scan
AllowNoIcons=yes
OutputDir=dist
OutputBaseFilename=DVR-Scan-win64-setup
SetupIconFile=dist\installer\dvr_scan_icon.ico
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
ArchitecturesAllowed=x64compatible
ChangesEnvironment=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "addtopath"; Description: "Add DVR-Scan to PATH"; GroupDescription: "Additional options:"; Flags: unchecked

[Files]
Source: "dist\dvr-scan\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\DVR-Scan CLI"; Filename: "{app}\dvr-scan.exe"
Name: "{group}\DVR-Scan GUI"; Filename: "{app}\dvr-scan-app.exe"
Name: "{group}\Uninstall DVR-Scan"; Filename: "{uninstallexe}"

[Registry]
Root: HKLM; Subkey: "SYSTEM\CurrentControlSet\Control\Session Manager\Environment"; ValueType: expandsz; ValueName: "Path"; ValueData: "{olddata};{app}"; Tasks: addtopath; Check: NeedsAddPath(ExpandConstant('{app}'))

[Code]
function NeedsAddPath(Param: string): boolean;
var
  OrigPath: string;
begin
  if not RegQueryStringValue(HKEY_LOCAL_MACHINE,
    'SYSTEM\CurrentControlSet\Control\Session Manager\Environment',
    'Path', OrigPath)
  then begin
    Result := True;
    exit;
  end;
  Result := Pos(';' + Param + ';', ';' + OrigPath + ';') = 0;
end;
