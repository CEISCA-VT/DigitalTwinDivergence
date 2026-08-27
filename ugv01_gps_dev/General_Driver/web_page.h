const char index_html[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
    <title>UGV01_BASE_WEB</title>
    <meta name="viewport" content="width=device-width,initial-scale=1.0">
    <!-- <script src="http://code.jquery.com/jquery-1.9.1.min.js"></script> -->
    <style type="text/css">
    html {
        display: inline-block;
        text-align: center;
        font-family: sans-serif;
    }
    body {
        background-image: -webkit-linear-gradient(#3F424F, #1E212E);
        font-family: "roboto",helt "sans-serif";
        font-weight: lighter;
        background-position: center 0;
        background-attachment: fixed;
        color: rgba(255, 255, 255, 0.6);
        font-size: 14px;
    }
    .cc-btn {
        border: 0;
        cursor: pointer;
        color: #fff;
        background: rgba(164,169,186,0);
        font-size: 1em;
        width: 100px;
        height: 100px;
         -webkit-touch-callout: none;
        -webkit-user-select: none;
        -khtml-user-select: none;
        -moz-user-select: none;
        -ms-user-select: none;
        user-select: none; 
    }
    .cc-middle{
        width: 100px;
        height: 100px;
        border-radius: 50%;
        background-color: rgba(94,98,112,0.8);
    }
    .cc-btn:hover svg, .cc-middle:hover {
        opacity: 0.5;
    }
    .cc-btn:active svg, .cc-middle:hover{
        opacity: 0.5;
    }
    .controlor-c > div{
        width: 300px;
        height: 300px; 
        background-color: rgba(94,98,112,0.2);
        border-radius: 40px;
        box-shadow: 10px 10px 10px rgba(0,0,0,0.05);
        margin: auto;
    }
    .controlor-c > div > div{
        display: flex;
    }
    main {
        width: min(1600px, calc(100% - 24px));
        margin: 0 auto;
    }
    section{margin: 40px 0;}
    .for-move {
        display: grid;
        grid-template-columns: minmax(0, 1.7fr) minmax(360px, 0.8fr);
        gap: 28px;
        align-items: start;
    }
    .for-move-a, .for-move-b{
        flex: 1;
        margin: 0 20px;
    }
    .h2-tt {
        font-size: 2em;
        font-weight: normal;
        color: rgba(255, 255, 255, 0.8);
        text-transform: uppercase;
    }
    .info-device-box .info-box{display: flex;}
    .info-device-box .info-box{padding: 20px 0;}
    .num-box-big > div, .num-box-sma > div{flex: 1;}
    .num-box-big > div:first-child{border-right: 1px solid rgba(216,216,216,0.1);}
    .num-box-mid {
        flex-wrap: wrap;
        justify-content: space-between;
    }
    .num-box-mid div{
        width:33.3333%;
        margin: 20px 0;
    }
    .info-device-box .info-box > div > span {
        display: block;
    }
    .info-box {
        background-image: linear-gradient(to right, rgba(94, 98, 112, 0.3), rgba(75, 80, 95, 0.3)) ;
        margin: 20px auto;
        border: 1px solid rgba(216, 216, 216, 0.1);
        box-shadow: 10px 10px 10px rgba(0,0,0,0.05);
        border-radius: 4px;
        color: rgba(255,255,255,0.5);
    }
    .big-num{font-size: 3em;}
    .mid-num{font-size: 2em;}
    .sma-num{font-size: 1.2em;}
    .num-color{
        background-image: linear-gradient(rgba(255,255,255,1),rgba(255,255,255,0.5));
        background-clip: text;
        color:transparent;
        -webkit-background-clip: text;
        -moz-background-clip: text;
        -ms-background-clip: text;
        font-weight: 900;
        line-height: 1em;
        margin: 0.5em 0;
    }
    .num-color-red{
        background-image: linear-gradient(rgba(181,104,108,1),rgba(181,104,108,0.5));
        background-clip: text;
        color:transparent;
        -webkit-background-clip: text;
        -moz-background-clip: text;
        -ms-background-clip: text;
        font-weight: 900;
        line-height: 1em;
        margin: 0.5em 0;
    }
    .controlor > div {margin: 80px 0;}
    .json-cmd-info{
        display: flex;
        flex-wrap: wrap;
    }
    .json-cmd-info > div {
        width: 33.33333%;
        padding: 10px 0;
    }
    .json-cmd-info p{
        line-height: 30px;
        margin: 0;
    }
    .json-cmd-info p span {
        display: block;
        color: rgba(255,255,255,0.8);
    }
    .small-btn{
        color: rgba(255,255,255,0.8);
        background-color: #5E6270;
        border: none;
        height: 48px;
        border-radius: 4px;
    }
    .small-btn-active{
        background-color: rgba(38,152,234,0.1);
        color: #2698EA;
        border: 1px solid #2698EA;
        height: 48px;
        border-radius: 4px;
    }
    .feedb-p input{
        width: 100%;
        height: 46px;
        background-color: rgba(0,0,0,0);
        padding: 0 10px;
        border: 1px solid rgba(194,196,201,0.15);
        border-radius: 4px;
        color: rgba(255, 255, 255, 0.8);
        font-size: 1.2em;
        margin-right: 10px;
    }
    .control-speed > div {
        width: 290px;
        margin: auto;
    }
    .control-speed > div > div{display: flex;}
    .control-speed label {flex: 1;}
    .small-btn, .small-btn-active{
        width: 90px;
    }
    .feedb-p{ display: flex;}
    .fb-input-info{
        margin: 0 20px;
    }
    .fb-info {margin: 20px;}
    .fb-info > span{line-height: 2.4em;}
    .btn-send:hover, .small-btn:hover{background-color: #2698EA;}
    .btn-send:active, .small-btn:active{background-color: #1b87d4;}
    .w-btn{
        color: #2698EA;
        background: transparent;
        padding: 10px;
        border: none;
    }
    .w-btn:hover{color: #2698EA;}
    .w-btn:active{color: #1b87d4;}
    .telemetry-box {
        padding: 16px;
        max-height: 620px;
        overflow-y: auto;
    }
    .telemetry-section {
        margin-bottom: 16px;
    }
    .telemetry-section:last-child {
        margin-bottom: 0;
    }
    .telemetry-section h3 {
        color: rgba(255,255,255,0.9);
        font-size: 0.95em;
        margin: 0 0 8px 0;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .telemetry-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
        gap: 8px;
    }
    .telemetry-grid > div {
        background: rgba(0,0,0,0.16);
        border: 1px solid rgba(194,196,201,0.10);
        border-radius: 4px;
        padding: 8px;
        min-width: 0;
    }
    .telemetry-grid span {
        display: block;
        color: #2698EA;
        font-size: 1.05em;
        font-weight: 700;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .telemetry-grid small {
        display: block;
        color: rgba(255,255,255,0.55);
        font-size: 0.7em;
        margin-top: 3px;
    }
    @media screen and (min-width: 768px) and (max-width: 1200px){
        body{font-size: 16px;}
        main {
            width: 100%;
        }
        .for-move {
            display: block;
        }
        /* .controlor-c > div{width: 600px;height: 600px;}
        .cc-btn{width: 200px;height: 200px;} */
        .json-cmd-info{display: block;}
        .json-cmd-info p span{display: inline;}
        .json-cmd-info > div{
            display: flex;
            width: auto;
            padding: 20px;
            flex-wrap: wrap;
            justify-content: space-between;
        }
        .control-speed > div{width: 600px;}
        section{margin: 20px 0;}
    }
    @media screen and (min-width: 360px) and (max-width: 767px){
        main {
            width: 100%;
        }
        .for-move {
            display: block;
        }
        .json-cmd-info{display: block;}
        .json-cmd-info p span{display: inline;}
        .json-cmd-info > div{
            display: flex;
            width: auto;
            padding: 20px;
            flex-wrap: wrap;
            justify-content: space-between;
        }
        section{margin: 10px 0;}
        .info-box{margin: 10px auto;}
        .info-device-box .info-box{padding: 10px;}
        .num-box-mid div{margin: 10px 0;}
        .controlor-c > div{
            width: 270px;
            height: 270px;
        }
        .cc-btn{
            width: 90px;
            height: 90px;;
        }
        .big-num{font-size: 2em;}
        .controlor > div{margin: 40px 0;}
    }
    
    /* Final telemetry dashboard layout */
    main {
        width: min(1400px, calc(100% - 32px));
        margin: 0 auto;
    }
    .for-move {
        display: grid;
        grid-template-columns: minmax(0, 1.35fr) minmax(300px, 0.65fr);
        gap: 28px;
        align-items: start;
    }
    .for-move-a, .for-move-b {
        min-width: 0;
        margin: 0;
    }
    .info-device-box {
        width: 100%;
    }
    .telemetry-box {
        padding: 18px;
        margin: 0;
        max-height: none;
        overflow: visible;
        box-sizing: border-box;
    }
    .telemetry-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 14px;
        margin-bottom: 14px;
        flex-wrap: wrap;
    }
    .telemetry-header h3 {
        margin: 0;
        color: rgba(255,255,255,0.9);
        font-size: 1.05em;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .logging-controls {
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
    }
    .logging-controls button {
        color: rgba(255,255,255,0.9);
        background: #5E6270;
        border: none;
        min-height: 38px;
        padding: 0 12px;
        border-radius: 4px;
        cursor: pointer;
    }
    .logging-controls button:hover { background: #2698EA; }
    .logging-status {
        color: rgba(255,255,255,0.65);
        font-size: 0.82em;
        min-width: 120px;
        text-align: left;
    }
    .telemetry-sections {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 14px;
    }
    .telemetry-section {
        margin: 0;
        padding: 12px;
        box-sizing: border-box;
        background: rgba(0,0,0,0.10);
        border: 1px solid rgba(194,196,201,0.08);
        border-radius: 5px;
        min-width: 0;
    }
    .telemetry-section.gps-section,
    .telemetry-section.timing-section {
        grid-column: 1 / -1;
    }
    .telemetry-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(86px, 1fr));
        gap: 7px;
        min-width: 0;
    }
    .telemetry-grid > div {
        min-width: 0;
        padding: 8px 5px;
        box-sizing: border-box;
        text-align: center;
        background: rgba(0,0,0,0.12);
    }
    .telemetry-grid span {
        font-size: clamp(0.9em, 1.3vw, 1.08em);
        line-height: 1.18;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .telemetry-grid small {
        white-space: normal;
        line-height: 1.15;
        font-size: 0.64em;
    }
    @media screen and (max-width: 1050px) {
        .for-move { grid-template-columns: 1fr; }
        .telemetry-sections { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media screen and (max-width: 700px) {
        main { width: calc(100% - 16px); }
        .telemetry-sections { grid-template-columns: 1fr; }
        .telemetry-section.gps-section,
        .telemetry-section.timing-section { grid-column: auto; }
        .logging-controls { width: 100%; }
        .logging-controls button { flex: 1 1 auto; }
        .logging-status { width: 100%; }
    }

</style>
</head>
<body>
    <main>
        <section>
            <div>
                <h2 class="h2-tt" id="deviceInfo">Control Panel</h2>
            </div>
            <div class="for-move">
                <div class="for-move-a">
                    <div class="info-device-box">
                        <div class="info-box telemetry-box">
                            <div class="telemetry-header">
                                <h3>Live Telemetry</h3>
                                <div class="logging-controls">
                                    <button type="button" onclick="startLogging();">START LOGGING</button>
                                    <button type="button" onclick="stopLogging();">STOP LOGGING</button>
                                    <button type="button" onclick="downloadCSV();">DOWNLOAD CSV</button>
                                    <span class="logging-status" id="loggingStatus">NOT LOGGING</span>
                                </div>
                            </div>

                            <div class="telemetry-sections">
                                <div class="telemetry-section">
                                    <h3>Motion / IMU</h3>
                                    <div class="telemetry-grid">
                                        <div><span id="r">--</span><small>ROLL</small></div>
                                        <div><span id="p">--</span><small>PITCH</small></div>
                                        <div><span id="y">--</span><small>YAW</small></div>
                                        <div><span id="v">--</span><small>VELOCITY</small></div>
                                        <div><span id="L">--</span><small>LEFT SPEED</small></div>
                                        <div><span id="R">--</span><small>RIGHT SPEED</small></div>
                                        <div><span id="temp">--</span><small>TEMP °C</small></div>
                                    </div>
                                </div>

                                <div class="telemetry-section">
                                    <h3>Acceleration</h3>
                                    <div class="telemetry-grid">
                                        <div><span id="ax">--</span><small>AX</small></div>
                                        <div><span id="ay">--</span><small>AY</small></div>
                                        <div><span id="az">--</span><small>AZ</small></div>
                                    </div>
                                </div>

                                <div class="telemetry-section">
                                    <h3>Gyroscope</h3>
                                    <div class="telemetry-grid">
                                        <div><span id="gx">--</span><small>GX</small></div>
                                        <div><span id="gy">--</span><small>GY</small></div>
                                        <div><span id="gz">--</span><small>GZ</small></div>
                                    </div>
                                </div>

                                <div class="telemetry-section">
                                    <h3>Magnetometer</h3>
                                    <div class="telemetry-grid">
                                        <div><span id="mx">--</span><small>MX</small></div>
                                        <div><span id="my">--</span><small>MY</small></div>
                                        <div><span id="mz">--</span><small>MZ</small></div>
                                    </div>
                                </div>

                                <div class="telemetry-section gps-section">
                                    <h3>GPS</h3>
                                    <div class="telemetry-grid gps-grid">
                                        <div><span id="gps_valid">--</span><small>VALID</small></div>
                                        <div><span id="gps_fix_type">--</span><small>FIX TYPE</small></div>
                                        <div><span id="gps_lat">--</span><small>LATITUDE</small></div>
                                        <div><span id="gps_lon">--</span><small>LONGITUDE</small></div>
                                        <div><span id="gps_sat">--</span><small>SATELLITES</small></div>
                                        <div><span id="gps_hdop">--</span><small>HDOP</small></div>
                                        <div><span id="gps_alt_m">--</span><small>ALTITUDE m</small></div>
                                        <div><span id="gps_speed_mps">--</span><small>GPS SPEED m/s</small></div>
                                        <div><span id="gps_course_deg">--</span><small>COURSE °</small></div>
                                        <div><span id="gps_age_ms">--</span><small>GPS AGE ms</small></div>
                                    </div>
                                </div>

                                <div class="telemetry-section timing-section">
                                    <h3>Encoders / Timing</h3>
                                    <div class="telemetry-grid">
                                        <div><span id="enc_left">--</span><small>ENC LEFT</small></div>
                                        <div><span id="enc_right">--</span><small>ENC RIGHT</small></div>
                                        <div><span id="sample_ms">--</span><small>SAMPLE ms</small></div>
                                        <div><span id="seq">--</span><small>SEQ</small></div>
                                        <div><span id="send_ms">--</span><small>SEND ms</small></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                <div class="for-move-b controlor">
                    <div class="controlor-c">
                        <div>
                            <div>
                                <label><button class="cc-btn" onmousedown="movtionButton(0.3,0.5);" ontouchstart="movtionButton(0.3,0.5);" onmouseup="movtionButton(0,0);" ontouchend="movtionButton(0,0);"><svg fill="none" version="1.1" width="23" height="23" viewBox="0 0 23 23"><g style="mix-blend-mode:passthrough"><path d="M0,2L0,18.1716C0,19.9534,2.15428,20.8457,3.41421,19.5858L19.5858,3.41421C20.8457,2.15428,19.9534,0,18.1716,0L2,0C0.895431,0,0,0.895431,0,2Z" fill="#D8D8D8" fill-opacity="0.20000000298023224"/></g></svg></button></label>
                                <label><button class="cc-btn" onmousedown="movtionButton(0.5,0.5);" ontouchstart="movtionButton(0.5,0.5);" onmouseup="movtionButton(0,0);" ontouchend="movtionButton(0,0);"><svg fill="none" version="1.1" width="26.87807685863504" height="15.435028109846826" viewBox="0 0 26.87807685863504 15.435028109846826"><g style="mix-blend-mode:passthrough" transform="matrix(0.9999999403953552,0,0,0.9999999403953552,0,0)"><path d="M12.0248,0.585787L0.589796,12.0208C-0.670133,13.2807,0.222199,15.435,2.00401,15.435L24.8741,15.435C26.6559,15.435,27.5482,13.2807,26.2883,12.0208L14.8533,0.585787C14.0722,-0.195262,12.8059,-0.195262,12.0248,0.585787Z" fill="#D8D8D8" fill-opacity="0.800000011920929"/></g></svg></button></label>
                                <label><button class="cc-btn" onmousedown="movtionButton(0.5,0.3);" ontouchstart="movtionButton(0.5,0.3);" onmouseup="movtionButton(0,0);" ontouchend="movtionButton(0,0);"><svg fill="none" version="1.1" width="23" height="23" viewBox="0 0 23 23"><g style="mix-blend-mode:passthrough" transform="matrix(0,1,-1,0,23,-23)"><path d="M23,2L23,18.1716C23,19.9534,25.15428,20.8457,26.41421,19.5858L42.5858,3.41421C43.8457,2.15428,42.9534,0,41.1716,0L25,0C23.895431,0,23,0.895431,23,2Z" fill="#D8D8D8" fill-opacity="0.20000000298023224"/></g></svg></button></label>
                            </div>
                            <div>
                                <label><button class="cc-btn" onmousedown="movtionButton(-0.5,0.5);" ontouchstart="movtionButton(-0.5,0.5);" onmouseup="movtionButton(0,0);" ontouchend="movtionButton(0,0);"><svg fill="none" version="1.1" width="15.435028109846769" height="26.87807685863504" viewBox="0 0 15.435028109846769 26.87807685863504"><g style="mix-blend-mode:passthrough" transform="matrix(0.9999999403953552,0,0,0.9999999403953552,0,0)"><path d="M0.585787,14.8533L12.0208,26.2883C13.2807,27.5482,15.435,26.6559,15.435,24.8741L15.435,2.00401C15.435,0.222199,13.2807,-0.670133,12.0208,0.589795L0.585787,12.0248C-0.195262,12.8059,-0.195262,14.0722,0.585787,14.8533Z" fill="#D8D8D8" fill-opacity="0.800000011920929"/></g></svg></button></label>
                                <label><button class="cc-btn cc-middle" onmousedown="movtionButton(0,0);" ontouchstart="movtionButton(0,0);" onmouseup="movtionButton(0,0);" ontouchend="movtionButton(0,0);">STOP</button></label>
                                <label><button class="cc-btn" onmousedown="movtionButton(0.5,-0.5);" ontouchstart="movtionButton(0.5,-0.5);" onmouseup="movtionButton(0,0);" ontouchend="movtionButton(0,0);"><svg fill="none" version="1.1" width="15.435030017195288" height="26.87807685863504" viewBox="0 0 15.435030017195288 26.87807685863504"><g style="mix-blend-mode:passthrough" transform="matrix(0.9999999403953552,0,0,0.9999999403953552,0,0)"><path d="M14.8492,12.0248L3.41422,0.589796C2.15429,-0.670133,-9.53674e-7,0.222199,9.53674e-7,2.00401L9.53674e-7,24.8741C-9.53674e-7,26.6559,2.15429,27.5482,3.41421,26.2883L14.8492,14.8533C15.6303,14.0722,15.6303,12.8059,14.8492,12.0248Z" fill="#D8D8D8" fill-opacity="0.800000011920929"/></g></svg></button></label>
                            </div>
                            <div>
                                <label><button class="cc-btn" onmousedown="movtionButton(-0.3,-0.5);" ontouchstart="movtionButton(-0.3,-0.5);" onmouseup="movtionButton(0,0);" ontouchend="movtionButton(0,0);"><svg fill="none" version="1.1" width="23" height="23" viewBox="0 0 23 23"><g style="mix-blend-mode:passthrough" transform="matrix(0,-1,1,0,-23,23)"><path d="M0,25L0,41.1716C0,42.9534,2.15428,43.8457,3.41421,42.5858L19.5858,26.41421C20.8457,25.15428,19.9534,23,18.1716,23L2,23C0.895431,23,0,23.895431,0,25Z" fill="#D8D8D8" fill-opacity="0.20000000298023224"/></g></svg></button></label>
                                <label><button class="cc-btn" onmousedown="movtionButton(-0.5,-0.5);" ontouchstart="movtionButton(-0.5,-0.5);" onmouseup="movtionButton(0,0);" ontouchend="movtionButton(0,0);"><svg fill="none" version="1.1" width="26.87807685863504" height="15.435030017195231" viewBox="0 0 26.87807685863504 15.435030017195231"><g style="mix-blend-mode:passthrough" transform="matrix(0.9999999403953552,0,0,0.9999999403953552,0,0)"><path d="M14.8533,14.8492L26.2883,3.41422C27.5482,2.15429,26.6559,-9.53674e-7,24.8741,9.53674e-7L2.00401,9.53674e-7C0.222199,-9.53674e-7,-0.670133,2.15429,0.589795,3.41421L12.0248,14.8492C12.8059,15.6303,14.0722,15.6303,14.8533,14.8492Z" fill="#D8D8D8" fill-opacity="0.800000011920929"/></g></svg></button></label>
                                <label><button class="cc-btn" onmousedown="movtionButton(-0.5,-0.3);" ontouchstart="movtionButton(-0.5,-0.3);" onmouseup="movtionButton(0,0);" ontouchend="movtionButton(0,0);"><svg fill="none" version="1.1" width="23" height="23" viewBox="0 0 23 23"><g style="mix-blend-mode:passthrough" transform="matrix(-1,0,0,-1,46,46)"><path d="M23,25L23,41.1716C23,42.9534,25.15428,43.8457,26.41421,42.5858L42.5858,26.41421C43.8457,25.15428,42.9534,23,41.1716,23L25,23C23.895431,23,23,23.895431,23,25Z" fill="#D8D8D8" fill-opacity="0.20000000298023224"/></g></svg></button></label>
                            </div>
                        </div>
                    </div>
                    <div class="control-speed">
                        <div>
                            <div id="device-gimbal-btn_A">
                                <label><button name="speedbtn" class="small-btn" onmousedown="gimbalCtrl(1);" ontouchstart="gimbalCtrl(1);" onmouseup="gimbalCtrl(0);" ontouchend="gimbalCtrl(0);">UP</button></label>
                            </div>
                        </div>
                        <br>
                        <div>
                            <div id="device-gimbal-btn_B">
                                <label><button name="speedbtn" class="small-btn" onmousedown="gimbalCtrl(3);" ontouchstart="gimbalCtrl(3);" onmouseup="gimbalCtrl(0);" ontouchend="gimbalCtrl(0);">LEFT</button></label>
                                <label><button name="speedbtn" class="small-btn" onmousedown="gimbalCtrl(5);" ontouchstart="gimbalCtrl(5);" onmouseup="gimbalCtrl(0);" ontouchend="gimbalCtrl(0);">FOEWARD</button></label>
                                <label><button name="speedbtn" class="small-btn" onmousedown="gimbalCtrl(4);" ontouchstart="gimbalCtrl(4);" onmouseup="gimbalCtrl(0);" ontouchend="gimbalCtrl(0);">RIGHT</button></label>
                            </div>
                        </div>
                        <br>
                        <div>
                            <div id="device-gimbal-btn_C">
                                <label><button name="speedbtn" class="small-btn" onmousedown="gimbalCtrl(2);" ontouchstart="gimbalCtrl(2);" onmouseup="gimbalCtrl(0);" ontouchend="gimbalCtrl(0);">DOWN</button></label>
                            </div>
                        </div>
                        <br>
                        <div>
                            <div id="device-gimbal-btn_D">
                                <label><button name="speedbtn" class="small-btn" onclick="gimbalSteady(1,read_Y);">STEADY START</button></label>
                                <label><button name="speedbtn" class="small-btn" onclick="gimbalSteady(0,read_Y);">STEADY END</button></label>
                            </div>
                        </div>
                        <br>
                        <br>
                        <div>
                            <div id="device-speed-btn">
                                <label><button name="speedbtn" class="small-btn" onclick="changeSpeed(0.5);">SLOW</button></label>
                                <label><button name="speedbtn" class="small-btn" onclick="changeSpeed(0.8);">MIDDLE</button></label>
                                <label><button name="speedbtn" class="small-btn" onclick="changeSpeed(1.0);">FAST</button></label>
                            </div>
                        </div>
                        <br>
                        <div>
                            <div id="device-led-btn">
                                <label><button name="speedbtn" class="small-btn" onclick="ledCtrl(1);">IO4</button></label>
                                <label><button name="speedbtn" class="small-btn" onclick="ledCtrl(2);">IO5</button></label>
                                <label><button name="speedbtn" class="small-btn" onclick="ledCtrl(0);">OFF</button></label>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </section>
        <section>
            <div class="fb-info">
                <h2 class="h2-tt" id="deviceInfo">Feedback information</h2>
                <span id="fbInfo" word-wrap="break-all">Json feedback information shows here.</span>
            </div>
            <div class="fb-input-info">
                <div class="feedb-p">
                    <input type="text" id="jsonData" placeholder="Input json cmd here.">
                    <label><button class="small-btn btn-send" onclick="jsonSend();">SEND</button></label>
                </div>
                <div class="info-box json-cmd-info">
                    <div>
                        <p>SPEED_CTRL: <span id="cmd1" class="cmd-value">{"T":1,"L":0.5,"R":0.5}</span></p>
                        <button class="w-btn" onclick="cmdFill('jsonData', 'cmd1');">INPUT</button>
                    </div>
                    <div>
                        <p>PWM_INPUT: <span id="cmd11" class="cmd-value">{"T":11,"L":164,"R":164}</span></p>
                        <button class="w-btn" onclick="cmdFill('jsonData', 'cmd11');">INPUT</button>
                    </div>
                    <div>
                        <p>ROS_CTRL: <span id="cmd13" class="cmd-value">{"T":13,"X":0.1,"Z":0.3}</span></p>
                        <button class="w-btn" onclick="cmdFill('jsonData', 'cmd13');">INPUT</button>
                    </div>
                    <div>
                        <p>PID_SET: <span id="cmd2" class="cmd-value">{"T":2,"P":200,"I":2500,"D":0,"L":255}</span></p>
                        <button class="w-btn" onclick="cmdFill('jsonData', 'cmd2');">INPUT</button>
                    </div>
                </div>
                <div class="info-box json-cmd-info">
                    <div>
                        <p>OLED_SET: <span id="cmd3" class="cmd-value">{"T":3,"lineNum":0,"Text":"putYourTextHere"}</span></p>
                        <button class="w-btn" onclick="cmdFill('jsonData', 'cmd3');">INPUT</button>
                    </div>
                    <div>
                        <p>OLED_DEFAULT: <span id="cmd-3" class="cmd-value">{"T":-3}</span></p>
                        <button class="w-btn" onclick="cmdFill('jsonData', 'cmd-3');">INPUT</button>
                    </div>
                </div>
                <div class="info-box json-cmd-info">
                    <div>
                        <p>CMD_MODULE_TYPE: <span id="cmd4" class="cmd-value">{"T":4,"cmd":0}</span></p>
                        <button class="w-btn" onclick="cmdFill('jsonData', 'cmd4');">INPUT</button>
                    </div>
                    <div>
                        <p>CMD_EOAT_TYPE: <span id="cmd124" class="cmd-value">{"T":124,"mode":0}</span></p>
                        <button class="w-btn" onclick="cmdFill('jsonData', 'cmd124');">INPUT</button>
                    </div>
                    <div>
                        <p>CMD_CONFIG_EOAT: <span id="cmd125" class="cmd-value">{"T":125,"pos":3,"ea":0,"eb":20}</span></p>
                        <button class="w-btn" onclick="cmdFill('jsonData', 'cmd125');">INPUT</button>
                    </div>
                </div>
                <div class="info-box json-cmd-info">
                    <div>
                        <p>CMD_GET_IMU_DATA: <span id="cmd126" class="cmd-value">{"T":126}</span></p>
                        <button class="w-btn" onclick="cmdFill('jsonData', 'cmd126');">INPUT</button>
                    </div>
                </div>
                <div class="info-box json-cmd-info">
                    <div>
                        <p>CMD_BASE_FEEDBACK: <span id="cmd130" class="cmd-value">{"T":130}</span></p>
                        <button class="w-btn" onclick="cmdFill('jsonData', 'cmd130');">INPUT</button>
                    </div>
                    <div>
                        <p>CMD_BASE_FEEDBACK_FLOW: <span id="cmd131" class="cmd-value">{"T":131,"cmd":0}</span></p>
                        <button class="w-btn" onclick="cmdFill('jsonData', 'cmd131');">INPUT</button>
                    </div>
                    <div>
                        <p>FEEDBACK_FLOW_INTERVAL: <span id="cmd142" class="cmd-value">{"T":142,"cmd":0}</span></p>
                        <button class="w-btn" onclick="cmdFill('jsonData', 'cmd142');">INPUT</button>
                    </div>
                    <div>
                        <p>CMD_UART_ECHO_MODE: <span id="cmd143" class="cmd-value">{"T":143,"cmd":0}</span></p>
                        <button class="w-btn" onclick="cmdFill('jsonData', 'cmd143');">INPUT</button>
                    </div>
                    <div>
                        <p>CMD_HEART_BEAT_SET: <span id="cmd136" class="cmd-value">{"T":136,"cmd":0}</span></p>
                        <button class="w-btn" onclick="cmdFill('jsonData', 'cmd136');">INPUT</button>
                    </div>
                </div>
                <div class="info-box json-cmd-info">
                    <div>
                        <p>CMD_LED_CTRL: <span id="cmd132" class="cmd-value">{"T":132,"IO4":255,"IO5":255}</span></p>
                        <button class="w-btn" onclick="cmdFill('jsonData', 'cmd132');">INPUT</button>
                    </div>
                    <div>
                        <p>CMD_GIMBAL_CTRL_SIMPLE: <span id="cmd133" class="cmd-value">{"T":133,"X":45,"Y":45,"SPD":0,"ACC":0}</span></p>
                        <button class="w-btn" onclick="cmdFill('jsonData', 'cmd133');">INPUT</button>
                    </div>
                    <div>
                        <p>CMD_GIMBAL_CTRL_MOVE: <span id="cmd134" class="cmd-value">{"T":134,"X":45,"Y":45,"SX":300,"SY":300}</span></p>
                        <button class="w-btn" onclick="cmdFill('jsonData', 'cmd134');">INPUT</button>
                    </div>
                    <div>
                        <p>CMD_GIMBAL_CTRL_STOP: <span id="cmd135" class="cmd-value">{"T":135}</span></p>
                        <button class="w-btn" onclick="cmdFill('jsonData', 'cmd135');">INPUT</button>
                    </div>
                    <div>
                        <p>CMD_GIMBAL_STEADY: <span id="cmd137" class="cmd-value">{"T":137,"s":1,"y":0}</span></p>
                        <button class="w-btn" onclick="cmdFill('jsonData', 'cmd137');">INPUT</button>
                    </div>
                    <div>
                        <p>CMD_GIMBAL_USER_CTRL: <span id="cmd141" class="cmd-value">{"T":141,"s":1,"y":0}</span></p>
                        <button class="w-btn" onclick="cmdFill('jsonData', 'cmd141');">INPUT</button>
                    </div>
                </div>
                <div class="info-box json-cmd-info">
                    <div>
                        <p>CMD_SET_SPD_RATE: <span id="cmd138" class="cmd-value">{"T":138,"L":1,"R":1}</span></p>
                        <button class="w-btn" onclick="cmdFill('jsonData', 'cmd138');">INPUT</button>
                    </div>
                    <div>
                        <p>CMD_GET_SPD_RATE: <span id="cmd139" class="cmd-value">{"T":139}</span></p>
                        <button class="w-btn" onclick="cmdFill('jsonData', 'cmd139');">INPUT</button>
                    </div>
                    <div>
                        <p>CMD_SAVE_SPD_RATE: <span id="cmd140" class="cmd-value">{"T":140}</span></p>
                        <button class="w-btn" onclick="cmdFill('jsonData', 'cmd140');">INPUT</button>
                    </div>
                </div>
                <div class="info-box json-cmd-info">
                    <div>
                        <p>CMD_MISSION_CONTENT: <span id="cmd221" class="cmd-value">{"T":221,"name":"mission_a"}</span></p>
                        <button class="w-btn" onclick="cmdFill('jsonData', 'cmd221');">INPUT</button>
                    </div>
                    <div>
                        <p>CMD_APPEND_STEP_JSON: <span id="cmd222" class="cmd-value">{"T":222,"name":"mission_a","step":"{\"T\":137,\"s\":1,\"y\":0}"}</span></p>
                        <button class="w-btn" onclick="cmdFill('jsonData', 'cmd222');">INPUT</button>
                    </div>
                </div>
                <div class="info-box json-cmd-info">
                    <div>
                        <p>CMD_BROADCAST_FOLLOWER: <span id="cmd300" class="cmd-value">{"T":300,"mode":1}</span></p>
                        <button class="w-btn" onclick="cmdFill('jsonData', 'cmd300');">INPUT</button>
                    </div>
                    <div>
                        <p>CMD_ESP_NOW_CONFIG: <span id="cmd301" class="cmd-value">{"T":301,"mode":3}</span></p>
                        <button class="w-btn" onclick="cmdFill('jsonData', 'cmd301');">INPUT</button>
                    </div>
                    <div>
                        <p>CMD_GET_MAC_ADDRESS: <span id="cmd302" class="cmd-value">{"T":302}</span></p>
                        <button class="w-btn" onclick="cmdFill('jsonData', 'cmd302');">INPUT</button>
                    </div>
                    <div>
                        <p>CMD_ESP_NOW_ADD_FOLLOWER: <span id="cmd303" class="cmd-value">{"T":303,"mac":"FF:FF:FF:FF:FF:FF"}</span></p>
                        <button class="w-btn" onclick="cmdFill('jsonData', 'cmd303');">INPUT</button>
                    </div>
                    <div>
                        <p>CMD_ESP_NOW_REMOVE_FOLLOWER: <span id="cmd304" class="cmd-value">{"T":304,"mac":"FF:FF:FF:FF:FF:FF"}</span></p>
                        <button class="w-btn" onclick="cmdFill('jsonData', 'cmd304');">INPUT</button>
                    </div>
                    <div>
                        <p>CMD_ESP_NOW_GROUP_CTRL: <span id="cmd305" class="cmd-value">{"T":305,"dev":0,"b":0,"s":0,"e":1.57,"h":1.57,"cmd":0,"megs":"hello!"}</span></p>
                        <button class="w-btn" onclick="cmdFill('jsonData', 'cmd305');">INPUT</button>
                    </div>
                    <div>
                        <p>CMD_ESP_NOW_SINGLE: <span id="cmd306" class="cmd-value">{"T":306,"mac":"FF:FF:FF:FF:FF:FF","dev":0,"b":0,"s":0,"e":1.57,"h":1.57,"cmd":0,"megs":"hello!"}</span></p>
                        <button class="w-btn" onclick="cmdFill('jsonData', 'cmd306');">INPUT</button>
                    </div>
                </div>
                <div class="info-box json-cmd-info">
                    <div>
                        <p>CMD_WIFI_ON_BOOT: <span id="cmd401" class="cmd-value">{"T":401,"cmd":3}</span></p>
                        <button class="w-btn" onclick="cmdFill('jsonData', 'cmd401');">INPUT</button>
                    </div>
                    <div>
                        <p>CMD_SET_AP: <span id="cmd402" class="cmd-value">{"T":402,"ssid":"UGV","password":"12345678"}</span></p>
                        <button class="w-btn" onclick="cmdFill('jsonData', 'cmd402');">INPUT</button>
                    </div>
                    <div>
                        <p>CMD_SET_STA: <span id="cmd403" class="cmd-value">{"T":403,"ssid":"na","password":"ps"}</span></p>
                        <button class="w-btn" onclick="cmdFill('jsonData', 'cmd403');">INPUT</button>
                    </div>
                    <div>
                        <p>CMD_WIFI_APSTA: <span id="cmd404" class="cmd-value">{"T":404,"ap_ssid":"UGV","ap_password":"12345678","sta_ssid":"na","sta_password":"ps"}</span></p>
                        <button class="w-btn" onclick="cmdFill('jsonData', 'cmd404');">INPUT</button>
                    </div>
                    <div>
                        <p>CMD_WIFI_INFO: <span id="cmd405" class="cmd-value">{"T":405}</span></p>
                        <button class="w-btn" onclick="cmdFill('jsonData', 'cmd405');">INPUT</button>
                    </div>
                    <div>
                        <p>CMD_WIFI_CONFIG_CREATE_BY_STATUS: <span id="cmd406" class="cmd-value">{"T":406}</span></p>
                        <button class="w-btn" onclick="cmdFill('jsonData', 'cmd406');">INPUT</button>
                    </div>
                    <div>
                        <p>CMD_WIFI_CONFIG_CREATE_BY_INPUT: <span id="cmd406" class="cmd-value">{"T":407,"mode":3,"ap_ssid":"UGV","ap_password":"12345678","sta_ssid":"na","sta_password":"ps"}</span></p>
                        <button class="w-btn" onclick="cmdFill('jsonData', 'cmd406');">INPUT</button>
                    </div>
                    <div>
                        <p>CMD_WIFI_STOP: <span id="cmd408" class="cmd-value">{"T":408}</span></p>
                        <button class="w-btn" onclick="cmdFill('jsonData', 'cmd408');">INPUT</button>
                    </div>
                </div>
                <div class="info-box json-cmd-info">
                    <div>
                        <p>CMD_SET_SERVO_ID: <span id="cmd501" class="cmd-value">{"T":501,"raw":1,"new":11}</span></p>
                        <button class="w-btn" onclick="cmdFill('jsonData', 'cmd501');">INPUT</button>
                    </div>
                    <div>
                        <p>CMD_SET_MIDDLE: <span id="cmd502" class="cmd-value">{"T":502,"id":11}</span></p>
                        <button class="w-btn" onclick="cmdFill('jsonData', 'cmd502');">INPUT</button>
                    </div>
                    <div>
                        <p>CMD_SET_SERVO_PID: <span id="cmd503" class="cmd-value">{"T":503,"id":14,"p":16}</span></p>
                        <button class="w-btn" onclick="cmdFill('jsonData', 'cmd503');">INPUT</button>
                    </div>
                </div>
                <div class="info-box json-cmd-info">
                    <div>
                        <p>CMD_REBOOT: <span id="cmd600" class="cmd-value">{"T":600}</span></p>
                        <button class="w-btn" onclick="cmdFill('jsonData', 'cmd600');">INPUT</button>
                    </div>
                    <div>
                        <p>CMD_FREE_FLASH_SPACE: <span id="cmd601" class="cmd-value">{"T":601}</span></p>
                        <button class="w-btn" onclick="cmdFill('jsonData', 'cmd601');">INPUT</button>
                    </div>
                    <div>
                        <p>CMD_BOOT_MISSION_INFO: <span id="cmd602" class="cmd-value">{"T":602}</span></p>
                        <button class="w-btn" onclick="cmdFill('jsonData', 'cmd602');">INPUT</button>
                    </div>
                    <div>
                        <p>CMD_RESET_BOOT_MISSION: <span id="cmd603" class="cmd-value">{"T":603}</span></p>
                        <button class="w-btn" onclick="cmdFill('jsonData', 'cmd603');">INPUT</button>
                    </div>
                    <div>
                        <p>CMD_NVS_CLEAR: <span id="cmd604" class="cmd-value">{"T":604}</span></p>
                        <button class="w-btn" onclick="cmdFill('jsonData', 'cmd604');">INPUT</button>
                    </div>
                    <div>
                        <p>CMD_INFO_PRINT: <span id="cmd605" class="cmd-value">{"T":605,"cmd":1}</span></p>
                        <button class="w-btn" onclick="cmdFill('jsonData', 'cmd605');">INPUT</button>
                    </div>
                </div>
                <div class="info-box json-cmd-info">
                    <div>
                        <p>CMD_MM_TYPE_SET: <span id="cmd900" class="cmd-value">{"T":900,"main":1,"module":0}</span></p>
                        <button class="w-btn" onclick="cmdFill('jsonData', 'cmd900');">INPUT</button>
                    </div>
                </div>
            </div>
        </section>
    </main>
<script>
    var cmdA;
    var cmdB;
    var cmdC;

    var forwardButton;   // 1
    var backwardButton;  // 2
    var fbNewer;

    var leftButton;      // 1
    var rightButton;     // 2
    var lrNewer;

    var last_forwardButton;   // 1
    var last_backwardButton;  // 2
    var last_fbNewer;

    var last_leftButton;      // 1
    var last_rightButton;     // 2
    var last_lrNewer;

    var speed_rate  = 1; // 1:fast 0.6:middle 0.3:slow
    var left_speed  = 0;
    var right_speed = 0;
    var send_heartbeat = 0;

    var io4_status = 0;
    var io5_status = 0;

    var gimbal_T = 135;
    var gimbal_X = 0;
    var gimbal_Y = 0;
    var read_X = 0;
    var read_Y = 0;

    var steady_status = 0;
    var steady_bias   = 0;

    var telemetryLogging = false;
    var telemetryRows = [];
    var telemetrySampleCount = 0;

    const telemetryColumns = [
        "pc_time", "T", "seq", "sample_ms", "millis", "L", "R",
        "enc_left", "enc_right", "v", "r", "p", "y",
        "ax", "ay", "az", "gx", "gy", "gz", "mx", "my", "mz",
        "temp", "gps_valid", "gps_age_ms", "gps_fix_type",
        "gps_lat", "gps_lon", "gps_sat", "gps_hdop", "gps_alt_m",
        "gps_speed_mps", "gps_course_deg", "gps_chars",
        "gps_sentences", "gps_failed_checksums", "lat", "lon", "sat",
        "hdop", "send_ms"
    ];

    ledCtrl(0);

    setInterval(function() {
        heartBeat();
    }, 1500);

    // Poll the complete telemetry packet at 10 Hz.
    // The existing ESP32 WebServer endpoint returns the JSON produced by T=147.
    var telemetryBusy = false;
    setInterval(function() {
        infoUpdate();
    }, 100);

    function cmdFill(rawInfo, fillInfo) {
        document.getElementById(rawInfo).value = document.getElementById(fillInfo).innerHTML;
    }
    function jsonSend() {
        send_heartbeat = 0;
        var xhttp = new XMLHttpRequest();
        xhttp.onreadystatechange = function() {
            if (this.readyState == 4 && this.status == 200) {
              document.getElementById("fbInfo").innerHTML =
              this.responseText;
            }
        };
        xhttp.open("GET", "js?json="+document.getElementById('jsonData').value, true);
        xhttp.send();
    }
    function telemetryValue(value, digits) {
        if (value === null || value === undefined || !Number.isFinite(Number(value))) {
            return "--";
        }
        return Number(value).toFixed(digits);
    }

    function telemetryInteger(value) {
        if (value === null || value === undefined || !Number.isFinite(Number(value))) {
            return "--";
        }
        return String(Math.trunc(Number(value)));
    }

    function telemetrySet(id, value, digits) {
        var element = document.getElementById(id);
        if (!element) return;
        element.innerHTML = (digits === undefined)
            ? telemetryInteger(value)
            : telemetryValue(value, digits);
    }

    function telemetrySetText(id, value) {
        var element = document.getElementById(id);
        if (!element) return;
        element.innerHTML = value === null || value === undefined ? "--" : value;
    }


    function csvEscape(value) {
        if (value === null || value === undefined) return "";
        var str = String(value);
        if (str.includes(",") || str.includes('"') || str.includes("\n") || str.includes("\r")) {
            return '"' + str.replace(/"/g, '""') + '"';
        }
        return str;
    }

    function telemetryCsvValue(obj, key) {
        return Object.prototype.hasOwnProperty.call(obj, key) ? obj[key] : "";
    }

    function logTelemetryToCsv(jsonResponse) {
        if (!telemetryLogging) return;

        var row = telemetryColumns.map(function(key) {
            if (key === "pc_time") return csvEscape(new Date().toISOString());
            return csvEscape(telemetryCsvValue(jsonResponse, key));
        });

        telemetryRows.push(row.join(","));
        telemetrySampleCount++;

        var status = document.getElementById("loggingStatus");
        if (status) {
            status.textContent = "LOGGING: " + telemetrySampleCount + " SAMPLES";
        }
    }

    function startLogging() {
        telemetryRows = [telemetryColumns.join(",")];
        telemetrySampleCount = 0;
        telemetryLogging = true;

        var status = document.getElementById("loggingStatus");
        if (status) status.textContent = "LOGGING: 0 SAMPLES";
    }

    function stopLogging() {
        telemetryLogging = false;

        var status = document.getElementById("loggingStatus");
        if (status) status.textContent = "STOPPED: " + telemetrySampleCount + " SAMPLES";
    }

    function downloadCSV() {
        if (telemetryRows.length <= 1) {
            alert("No telemetry data has been recorded.");
            return;
        }

        var csvContent = telemetryRows.join("\r\n");
        var blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
        var url = URL.createObjectURL(blob);
        var link = document.createElement("a");
        var stamp = new Date().toISOString().replace(/[:.]/g, "-");

        link.href = url;
        link.download = "UGV_telemetry_" + stamp + ".csv";
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    }

    function infoUpdate() {
        // Do not queue multiple HTTP requests if the ESP32 is still processing
        // the previous telemetry request.
        if (telemetryBusy) return;
        telemetryBusy = true;

        var jsonString = JSON.stringify({"T":147});
        var xhttp = new XMLHttpRequest();

        xhttp.onreadystatechange = function() {
            if (this.readyState == 4) {
                telemetryBusy = false;

                if (this.status == 200) {
                    try {
                        var jsonResponse = JSON.parse(this.responseText);

                        logTelemetryToCsv(jsonResponse);

                        telemetrySet("r", jsonResponse.r, 2);
                        telemetrySet("p", jsonResponse.p, 2);
                        telemetrySet("y", jsonResponse.y, 2);
                        telemetrySet("v", jsonResponse.v, 4);
                        telemetrySet("L", jsonResponse.L, 3);
                        telemetrySet("R", jsonResponse.R, 3);
                        telemetrySet("temp", jsonResponse.temp, 2);

                        telemetrySet("ax", jsonResponse.ax, 3);
                        telemetrySet("ay", jsonResponse.ay, 3);
                        telemetrySet("az", jsonResponse.az, 3);

                        telemetrySet("gx", jsonResponse.gx, 4);
                        telemetrySet("gy", jsonResponse.gy, 4);
                        telemetrySet("gz", jsonResponse.gz, 4);

                        telemetrySet("mx", jsonResponse.mx, 2);
                        telemetrySet("my", jsonResponse.my, 2);
                        telemetrySet("mz", jsonResponse.mz, 2);

                        telemetrySetText("gps_valid",
                            jsonResponse.gps_valid === true ? "YES" :
                            jsonResponse.gps_valid === false ? "NO" : "--");
                        telemetrySet("gps_fix_type", jsonResponse.gps_fix_type);
                        telemetrySet("gps_lat", jsonResponse.gps_lat, 6);
                        telemetrySet("gps_lon", jsonResponse.gps_lon, 6);
                        telemetrySet("gps_sat", jsonResponse.gps_sat);
                        telemetrySet("gps_hdop", jsonResponse.gps_hdop, 2);
                        telemetrySet("gps_alt_m", jsonResponse.gps_alt_m, 2);
                        telemetrySet("gps_speed_mps", jsonResponse.gps_speed_mps, 3);
                        telemetrySet("gps_course_deg", jsonResponse.gps_course_deg, 2);

                        // TinyGPS++ uses UINT32_MAX for an unavailable age.
                        if (jsonResponse.gps_age_ms === 4294967295) {
                            telemetrySetText("gps_age_ms", "N/A");
                        } else {
                            telemetrySet("gps_age_ms", jsonResponse.gps_age_ms);
                        }

                        telemetrySet("enc_left", jsonResponse.enc_left);
                        telemetrySet("enc_right", jsonResponse.enc_right);
                        telemetrySet("sample_ms", jsonResponse.sample_ms);
                        telemetrySet("seq", jsonResponse.seq);
                        telemetrySet("send_ms", jsonResponse.send_ms);
                    } catch (error) {
                        console.error("Invalid telemetry JSON:", error);
                    }
                }
            }
        };

        xhttp.onerror = function() {
            telemetryBusy = false;
        };

        xhttp.open("GET", "js?json=" + encodeURIComponent(jsonString), true);
        xhttp.send();
    }

    function changeSpeed(inputSpd) {
        speed_rate = inputSpd;
    }
    function heartBeat() {
        if (send_heartbeat == 1) {
            var jsonCmd = {
                "T":1,
                "L":left_speed,
                "R":right_speed
            }
            var jsonString = JSON.stringify(jsonCmd);
            var xhr = new XMLHttpRequest();
            xhr.open("GET", "js?json=" + jsonString, true);
            xhr.send();
        }
    }
    function movtionButton(spdL, spdR){
        left_speed  = spdL*speed_rate;
        right_speed = spdR*speed_rate;
        send_heartbeat = 1;
        var jsonCmd = {
            "T":1,
            "L":left_speed,
            "R":right_speed
        }
        var jsonString = JSON.stringify(jsonCmd);
        var xhr = new XMLHttpRequest();
        xhr.open("GET", "js?json=" + jsonString, true);
        xhr.send();
    }
    function ledCtrl(inputCmd){
        if (inputCmd == 0) {
            io4_status = 0;
            io5_status = 0;
        }
        else if (inputCmd == 1) {
            if (io4_status == 0) {
                io4_status = 255;
            }
            else {
                io4_status = 0;
            }
        }
        else if (inputCmd == 2) {
            if (io5_status == 0) {
                io5_status = 255;
            }
            else {
                io5_status = 0;
            }
        }
        var jsonCmd = {
            "T":132,
            "IO4":io4_status,
            "IO5":io5_status
        }
        var jsonString = JSON.stringify(jsonCmd);
        var xhr = new XMLHttpRequest();
        xhr.open("GET", "js?json=" + jsonString, true);
        xhr.send();
    }
    function gimbalSteady(inputS,inputY){
        steady_status = inputS;
        steady_bias = inputY;
        var jsonCmd = {
            "T":137,
            "s":steady_status,
            "y":steady_bias
        }
        var jsonString = JSON.stringify(jsonCmd);
        var xhr = new XMLHttpRequest();
        xhr.open("GET", "js?json=" + jsonString, true);
        xhr.send();
    }
    function gimbalCtrl(inputCmd){
        if (inputCmd == 0) {
            gimbal_T = 135;
        }else if (inputCmd == 1) {
            gimbal_T = 134;
            gimbal_X = read_X;
            gimbal_Y = 90;
            if (steady_status == 1) {
                steady_bias = steady_bias + 5;
                if (steady_bias > 90) {
                    steady_bias = 90;
                }
            }
        }else if (inputCmd == 2) {
            gimbal_T = 134;
            gimbal_X = read_X;
            gimbal_Y = -45;
            if (steady_status == 1) {
                steady_bias = steady_bias - 5;
                if (steady_bias < -45) {
                    steady_bias = 45;
                }
            }
        }else if (inputCmd == 3) {
            gimbal_T = 134;
            gimbal_X = -180;
            gimbal_Y = read_Y;
        }else if (inputCmd == 4) {
            gimbal_T = 134;
            gimbal_X = 180;
            gimbal_Y = read_Y;
        }else if (inputCmd == 5) {
            gimbal_T = 134;
            gimbal_X = 0;
            gimbal_Y = 0;
            if (steady_status == 1) {
                steady_bias = 0;
            }
        }

        if (steady_status == 0) {
            var jsonCmd = {
                "T":gimbal_T,
                "X":gimbal_X,
                "Y":gimbal_Y,
                "SX":600,
                "SY":600
            }
            var jsonString = JSON.stringify(jsonCmd);
            var xhr = new XMLHttpRequest();
            xhr.open("GET", "js?json=" + jsonString, true);
            xhr.send();
        }else if (steady_status == 1) {
            gimbalSteady(1,steady_bias);
        }
    }

    function cmdProcess(){
        if (forwardButton == 0 && backwardButton == 0 && leftButton == 0 && rightButton == 0) {
            movtionButton(0, 0);
        }
        else if (forwardButton == 1 && backwardButton == 0 && leftButton == 0 && rightButton == 0){
            movtionButton(0.5, 0.5);
        }else if (forwardButton == 1 && backwardButton == 1 && fbNewer == 1 && leftButton == 0 && rightButton == 0){
            movtionButton(0.5, 0.5);
        }else if (forwardButton == 1 && backwardButton == 1 && fbNewer == 2 && leftButton == 0 && rightButton == 0){
            movtionButton(-0.5, -0.5);
        }else if (forwardButton == 0 && backwardButton == 1 && leftButton == 0 && rightButton == 0){
            movtionButton(-0.5, -0.5);
        }
        else if (forwardButton == 0 && backwardButton == 0 && leftButton == 1 && rightButton == 0){
            movtionButton(-0.5, 0.5);
        }else if (forwardButton == 0 && backwardButton == 0 && leftButton == 1 && rightButton == 1 && lrNewer == 1){
            movtionButton(-0.5, 0.5);
        }else if (forwardButton == 0 && backwardButton == 0 && leftButton == 0 && rightButton == 1){
            movtionButton(0.5, -0.5);
        }else if (forwardButton == 0 && backwardButton == 0 && leftButton == 1 && rightButton == 1 && lrNewer == 2){
            movtionButton(0.5, -0.5);
        }
        else if (forwardButton == 1 && backwardButton == 0 && leftButton == 1 && rightButton == 0){
            movtionButton(0.3, 0.5);
        }else if (forwardButton == 1 && backwardButton == 1 && fbNewer == 1 && leftButton == 1 && rightButton == 0){
            movtionButton(0.3, 0.5);
        }else if (forwardButton == 1 && backwardButton == 1 && fbNewer == 1 && leftButton == 1 && rightButton == 1 && lrNewer == 1){
            movtionButton(0.3, 0.5);
        }else if (forwardButton == 1 && backwardButton == 0 && leftButton == 1 && rightButton == 1 && lrNewer == 1){
            movtionButton(0.3, 0.5);
        }
        else if (forwardButton == 1 && backwardButton == 0 && leftButton == 0 && rightButton == 1){
            movtionButton(0.5, 0.3);
        }else if (forwardButton == 1 && backwardButton == 1 && fbNewer == 1 && leftButton == 0 && rightButton == 1){
            movtionButton(0.5, 0.3);
        }else if (forwardButton == 1 && backwardButton == 1 && fbNewer == 1 && leftButton == 1 && rightButton == 1 && lrNewer == 2){
            movtionButton(0.5, 0.3);
        }else if (forwardButton == 1 && backwardButton == 0 && leftButton == 1 && rightButton == 1 && lrNewer == 2){
            movtionButton(0.5, 0.3);
        }
        else if (forwardButton == 0 && backwardButton == 1 && leftButton == 1 && rightButton == 0){
            movtionButton(-0.3, -0.5);
        }else if (forwardButton == 1 && backwardButton == 1 && fbNewer == 2 && leftButton == 1 && rightButton == 0){
            movtionButton(-0.3, -0.5);
        }else if (forwardButton == 1 && backwardButton == 1 && fbNewer == 2 && leftButton == 1 && rightButton == 1 && lrNewer == 1){
            movtionButton(-0.3, -0.5);
        }else if (forwardButton == 0 && backwardButton == 1 && leftButton == 1 && rightButton == 1 && lrNewer == 1){
            movtionButton(-0.3, -0.5);
        }
        else if (forwardButton == 0 && backwardButton == 1 && leftButton == 0 && rightButton == 1){
            movtionButton(-0.5, -0.3);
        }else if (forwardButton == 1 && backwardButton == 1 && fbNewer == 2 && leftButton == 0 && rightButton == 1){
            movtionButton(-0.5, -0.3);
        }else if (forwardButton == 1 && backwardButton == 1 && fbNewer == 2 && leftButton == 1 && rightButton == 1 && lrNewer == 2){
            movtionButton(-0.5, -0.3);
        }else if (forwardButton == 0 && backwardButton == 1 && leftButton == 1 && rightButton == 1 && lrNewer == 2){
            movtionButton(-0.5, -0.3);
        }
    }

    document.onkeydown = function (event) {
        var e = event || window.event || arguments.callee.caller.arguments[0];
        if (e && e.keyCode == 65) {
            // alert ("A down");
            leftButton = 1;
            lrNewer = 1;
        }else if (e && e.keyCode == 87) {
            // alert ("W down");
            forwardButton = 1;
            fbNewer = 1;
        }else if (e && e.keyCode == 83) {
            // alert ("S down");
            backwardButton = 1;
            fbNewer = 2;
        }else if (e && e.keyCode == 68) {
            // alert ("D down");
            rightButton = 1;
            lrNewer = 2;
        }
        else if (e && e.keyCode == 13) {
            // alert ("Enter down");
            jsonSend();
        }
        else if (e && e.keyCode == 37) {
            // alert ("left down");
            leftButton = 1;
            lrNewer = 1;
        }else if (e && e.keyCode == 38) {
            // alert ("up down");
            forwardButton = 1;
            fbNewer = 1;
        }else if (e && e.keyCode == 40) {
            // alert ("down down");
            backwardButton = 1;
            fbNewer = 2;
        }else if (e && e.keyCode == 39) {
            // alert ("right down");
            rightButton = 1;
            lrNewer = 2;
        }

        if(forwardButton != last_forwardButton || backwardButton != last_backwardButton || fbNewer != last_fbNewer || leftButton != last_leftButton || last_rightButton != rightButton || lrNewer != last_fbNewer) {
            cmdProcess();
            last_forwardButton = forwardButton;
            last_backwardButton = backwardButton;
            last_fbNewer = fbNewer;

            last_leftButton = leftButton;
            last_rightButton = rightButton;
            last_lrNewer = lrNewer;
        }
    }

    document.onkeyup = function (event) {
        var e = event || window.event || arguments.callee.caller.arguments[0];
        if (e && e.keyCode == 65) {
            // alert ("A up");
            leftButton = 0;
        }else if (e && e.keyCode == 87) {
            // alert ("W up");
            forwardButton = 0;
        }else if (e && e.keyCode == 83) {
            // alert ("S up");
            backwardButton = 0;
        }else if (e && e.keyCode == 68) {
            // alert ("D up");
            rightButton = 0;
        }
        else if (e && e.keyCode == 37) {
            // alert ("left up");
            leftButton = 0;
        }else if (e && e.keyCode == 38) {
            // alert ("up up");
            forwardButton = 0;
        }else if (e && e.keyCode == 40) {
            // alert ("down up");
            backwardButton = 0;
        }else if (e && e.keyCode == 39) {
            // alert ("right up");
            rightButton = 0;
        }

        cmdProcess();

        last_forwardButton = forwardButton;
        last_backwardButton = backwardButton;
        last_fbNewer = fbNewer;

        last_leftButton = leftButton;
        last_rightButton = rightButton;
        last_lrNewer = lrNewer;
    }
</script>
</body>
</html>
)rawliteral";
