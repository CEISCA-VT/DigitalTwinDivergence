#ifndef WEB_PAGE_H
#define WEB_PAGE_H

const char index_html[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>UGV01 JSON Control</title>
<style>
    * { box-sizing: border-box; }
    body {
        margin: 0;
        padding: 32px 16px;
        background: #111;
        color: #eee;
        font-family: Arial, Helvetica, sans-serif;
    }
    main {
        width: min(100%, 1000px);
        margin: 0 auto;
    }
    h1 { margin: 0 0 8px; font-size: 28px; }
    .subtitle {
        margin: 0 0 24px;
        color: #aaa;
    }
    .panel {
        border: 1px solid #333;
        border-radius: 8px;
        padding: 20px;
        background: #181818;
    }
    .json-row {
        display: flex;
        gap: 10px;
        align-items: stretch;
    }
    #jsonData {
        flex: 1;
        min-width: 0;
        padding: 13px;
        border: 1px solid #444;
        border-radius: 5px;
        background: #0d0d0d;
        color: #fff;
        font: 15px Consolas, "Courier New", monospace;
    }
    button {
        border: 0;
        border-radius: 5px;
        padding: 0 20px;
        background: #ddd;
        color: #111;
        font-weight: bold;
        cursor: pointer;
    }
    button:hover { background: #fff; }
    .controls {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-top: 14px;
    }
    .stream-toggle {
        position: relative;
        width: 52px;
        height: 28px;
        flex: 0 0 auto;
    }
    .stream-toggle input {
        opacity: 0;
        width: 0;
        height: 0;
    }
    .slider {
        position: absolute;
        inset: 0;
        border-radius: 28px;
        background: #444;
        cursor: pointer;
        transition: .2s;
    }
    .slider:before {
        content: "";
        position: absolute;
        width: 22px;
        height: 22px;
        left: 3px;
        top: 3px;
        border-radius: 50%;
        background: #fff;
        transition: .2s;
    }
    input:checked + .slider { background: #4caf50; }
    input:checked + .slider:before { transform: translateX(24px); }
    #streamStatus {
        color: #aaa;
        font-size: 14px;
    }
    #response {
        margin-top: 14px;
        min-height: 20px;
        color: #aaa;
        font: 13px Consolas, "Courier New", monospace;
        white-space: pre-wrap;
        word-break: break-all;
    }
    .reference {
        margin-top: 28px;
    }
    .reference h2 {
        font-size: 19px;
        margin: 0 0 8px;
    }
    .reference p {
        margin: 0 0 10px;
        color: #aaa;
    }
    #commandReference {
        margin: 0;
        padding: 16px;
        border: 1px solid #333;
        border-radius: 5px;
        background: #0d0d0d;
        color: #ccc;
        overflow-x: auto;
        white-space: pre-wrap;
        font: 13px/1.55 Consolas, "Courier New", monospace;
    }
    @media (max-width: 650px) {
        .json-row { flex-direction: column; }
        #jsonData { min-height: 90px; }
        button { min-height: 44px; }
    }
</style>
</head>
<body>
<main>
    <h1>UGV01 JSON Control</h1>
    <p class="subtitle">Send JSON commands directly to the vehicle.</p>

    <section class="panel">
        <div class="json-row">
            <input
                type="text"
                id="jsonData"
                placeholder='{"T":1,"L":0.5,"R":0.5}'
                autocomplete="off"
                spellcheck="false">
            <button type="button" onclick="jsonSend()">SEND</button>
        </div>

        <div class="controls">
            <label class="stream-toggle" title="Enable or disable telemetry streaming">
                <input type="checkbox" id="streamToggle" onchange="toggleStreaming()">
                <span class="slider"></span>
            </label>
            <span id="streamStatus">Streaming disabled</span>
        </div>

        <div id="response"></div>
    </section>

    <section class="reference">
        <h2>JSON Command Reference</h2>
        <p>Command examples and their purpose. Copy an example into the JSON input above and edit its values as needed.</p>
        <pre id="commandReference">LATENCY_TEST (T=998)         Test computer ↔ UGV network latency.
Example: {"T":998}
The browser reports HTTP round-trip time. No vehicle motion occurs.

SPEED_CTRL (T=1)              Set left/right UGV speed.
Example: {"T":1,"L":0.5,"R":0.5}

PWM_INPUT (T=11)              Set motor PWM directly.
Example: {"T":11,"L":164,"R":164}

ROS_CTRL (T=13)               Set linear/angular velocity.
Example: {"T":13,"X":0.1,"Z":0.3}

PID_SET (T=2)                 Set motor PID parameters.
Example: {"T":2,"P":200,"I":2500,"D":0,"L":255}

OLED_SET (T=3)                Set one OLED display line.
Example: {"T":3,"lineNum":0,"Text":"putYourTextHere"}

OLED_DEFAULT (T=-3)           Restore the default OLED display.
Example: {"T":-3}

MODULE_TYPE (T=4)             Set the attached module type.
Example: {"T":4,"cmd":0}

EOAT_TYPE (T=124)             Select gripper/wrist end-effector type.
Example: {"T":124,"mode":0}

CONFIG_EOAT (T=125)           Configure end-effector mounting.
Example: {"T":125,"pos":3,"ea":0,"eb":20}

GET_IMU_DATA (T=126)           Request IMU data.
Example: {"T":126}

BASE_FEEDBACK (T=130)          Request base feedback.
Example: {"T":130}

BASE_FEEDBACK_FLOW (T=131)     Enable/disable base feedback flow.
Example: {"T":131,"cmd":0}

FEEDBACK_FLOW_INTERVAL (T=142) Set feedback flow interval.
Example: {"T":142,"cmd":0}

UART_ECHO_MODE (T=143)         Enable/disable UART command echo.
Example: {"T":143,"cmd":0}

GET_GPS_DATA (T=146)           Request GPS data.
Example: {"T":146}

GET_ALL_TELEMETRY (T=147)      Request complete telemetry packet.
Example: {"T":147}

LED_CTRL (T=132)               Control IO4/IO5 LEDs.
Example: {"T":132,"IO4":255,"IO5":255}

GIMBAL_CTRL_SIMPLE (T=133)     Move gimbal to a target.
Example: {"T":133,"X":45,"Y":45,"SPD":0,"ACC":0}

GIMBAL_CTRL_MOVE (T=134)       Move gimbal with speed settings.
Example: {"T":134,"X":45,"Y":45,"SX":300,"SY":300}

GIMBAL_CTRL_STOP (T=135)       Stop gimbal motion.
Example: {"T":135}

HEART_BEAT_SET (T=136)         Set heartbeat delay.
Example: {"T":136,"cmd":3000}

GIMBAL_STEADY (T=137)          Enable/disable gimbal steady mode.
Example: {"T":137,"s":1,"y":0}

SET_SPD_RATE (T=138)           Set UGV speed rate.
Example: {"T":138,"L":1,"R":1}

GET_SPD_RATE (T=139)            Get UGV speed rate.
Example: {"T":139}

SAVE_SPD_RATE (T=140)           Save UGV speed rate.
Example: {"T":140}

GIMBAL_USER_CTRL (T=141)       User gimbal control.
Example: {"T":141,"X":0,"Y":0,"SPD":300}

MOVE_INIT (T=100)               Move arm to initialization position.
Example: {"T":100}

SINGLE_JOINT_CTRL (T=101)      Control one arm joint in radians.
Example: {"T":101,"joint":1,"rad":0,"spd":0,"acc":10}

JOINTS_RAD_CTRL (T=102)        Control all arm joints in radians.
Example: {"T":102,"base":0,"shoulder":0,"elbow":1.57,"hand":1.57,"spd":0,"acc":10}

SINGLE_AXIS_CTRL (T=103)       Control one arm Cartesian axis.
Example: {"T":103,"axis":2,"pos":0,"spd":0.25}

XYZT_GOAL_CTRL (T=104)         Move arm to an XYZ/T goal.
Example: {"T":104,"x":235,"y":0,"z":234,"t":3.14,"spd":0.25}

XYZT_DIRECT_CTRL (T=1041)      Direct XYZ/T arm control.
Example: {"T":1041,"x":235,"y":0,"z":234,"t":3.14}

SERVO_RAD_FEEDBACK (T=105)     Request arm servo feedback.
Example: {"T":105}

EOAT_HAND_CTRL (T=106)         Control the hand/end-effector angle.
Example: {"T":106,"cmd":1.57,"spd":0,"acc":0}

EOAT_GRAB_TORQUE (T=107)       Set end-effector grab torque.
Example: {"T":107,"tor":200}

SET_JOINT_PID (T=108)          Set a joint P/I controller.
Example: {"T":108,"joint":3,"p":16,"i":0}

RESET_PID (T=109)              Reset arm PID settings.
Example: {"T":109}

SET_NEW_X (T=110)              Set a new arm X-axis reference.
Example: {"T":110,"xAxisAngle":0}

DELAY_MILLIS (T=111)           Delay for the specified milliseconds.
Example: {"T":111,"cmd":3000}

DYNAMIC_ADAPTATION (T=112)    Set dynamic external-force torque limits.
Example: {"T":112,"mode":1,"b":60,"s":110,"e":50,"h":50}

SWITCH_CTRL (T=113)            Control the 12V switch output.
Example: {"T":113,"pwm_a":-255,"pwm_b":-255}

LIGHT_CTRL (T=114)             Control the light output.
Example: {"T":114,"led":255}

SWITCH_OFF (T=115)             Turn the switch output off.
Example: {"T":115}

SINGLE_JOINT_ANGLE (T=121)     Control one joint in degrees.
Example: {"T":121,"joint":1,"angle":0,"spd":10,"acc":10}

JOINTS_ANGLE_CTRL (T=122)      Control all joints in degrees.
Example: {"T":122,"b":0,"s":0,"e":90,"h":180,"spd":10,"acc":10}

CONSTANT_CTRL (T=123)           Constant arm control.
Example: {"T":123,"m":0,"axis":0,"cmd":0,"spd":3}

SCAN_FILES (T=200)             Scan files stored in flash.
Example: {"T":200}

CREATE_FILE (T=201)            Create a flash file.
Example: {"T":201,"name":"file.txt","content":"inputContentHere."}

READ_FILE (T=202)              Read a flash file.
Example: {"T":202,"name":"file.txt"}

DELETE_FILE (T=203)             Delete a flash file.
Example: {"T":203,"name":"file.txt"}

APPEND_LINE (T=204)            Append a line to a file.
Example: {"T":204,"name":"file.txt","content":"inputContentHere."}

INSERT_LINE (T=205)             Insert a line into a file.
Example: {"T":205,"name":"file.txt","lineNum":3,"content":"content"}

REPLACE_LINE (T=206)            Replace a line in a file.
Example: {"T":206,"name":"file.txt","lineNum":3,"content":"Content"}

READ_LINE (T=207)               Read one line from a file.
Example: {"T":207,"name":"file.txt","lineNum":3}

DELETE_LINE (T=208)             Delete one line from a file.
Example: {"T":208,"name":"file.txt","lineNum":3}

TORQUE_CTRL (T=210)             Enable/disable torque lock.
Example: {"T":210,"cmd":1}

CREATE_MISSION (T=220)          Create a mission in flash.
Example: {"T":220,"name":"mission_a","intro":"test mission created in flash."}

MISSION_CONTENT (T=221)         Get the contents of a mission.
Example: {"T":221,"name":"mission_a"}

APPEND_STEP_JSON (T=222)        Append a JSON step to a mission.
Example: {"T":222,"name":"mission_a","step":"{\"T\":104,\"x\":235,\"y\":0,\"z\":234,\"t\":3.14,\"spd\":0.25}"}

APPEND_STEP_FB (T=223)          Append a feedback step to a mission.
Example: {"T":223,"name":"mission_a","spd":0.25}

APPEND_DELAY (T=224)            Append a delay to a mission.
Example: {"T":224,"name":"mission_a","delay":3000}

INSERT_STEP_JSON (T=225)        Insert a JSON step into a mission.
Example: {"T":225,"name":"mission_a","stepNum":3,"step":"{\"T\":114,\"led\":255}"}

INSERT_STEP_FB (T=226)          Insert a feedback step into a mission.
Example: {"T":226,"name":"mission_a","stepNum":3,"spd":0.25}

INSERT_DELAY (T=227)            Insert a delay into a mission.
Example: {"T":227,"stepNum":3,"delay":3000}

REPLACE_STEP_JSON (T=228)       Replace a mission step with JSON.
Example: {"T":228,"name":"mission_a","stepNum":3,"step":"{\"T\":114,\"led\":255}"}

REPLACE_STEP_FB (T=229)         Replace a mission step with feedback.
Example: {"T":229,"name":"mission_a","stepNum":3,"spd":0.25}

REPLACE_DELAY (T=230)           Replace a mission step with a delay.
Example: {"T":230,"name":"mission_a","stepNum":3,"delay":3000}

DELETE_STEP (T=231)              Delete a mission step.
Example: {"T":231,"name":"mission_a","stepNum":3}

MOVE_TO_STEP (T=241)             Move execution to a mission step.
Example: {"T":241,"name":"mission_a","stepNum":3}

MISSION_PLAY (T=242)             Play a mission.
Example: {"T":242,"name":"mission_a","times":3}

BROADCAST_FOLLOWER (T=300)      Configure ESP-NOW follower behavior.
Example: {"T":300,"mode":1}

ESP_NOW_CONFIG (T=301)           Configure ESP-NOW mode.
Example: {"T":301,"mode":3}

GET_MAC_ADDRESS (T=302)          Get the device MAC address.
Example: {"T":302}

ESP_NOW_ADD_FOLLOWER (T=303)    Add an ESP-NOW follower.
Example: {"T":303,"mac":"FF:FF:FF:FF:FF:FF"}

ESP_NOW_REMOVE_FOLLOWER (T=304) Remove an ESP-NOW follower.
Example: {"T":304,"mac":"FF:FF:FF:FF:FF:FF"}

ESP_NOW_GROUP_CTRL (T=305)      Send group control.
Example: {"T":305,"dev":0,"b":0,"s":0,"e":1.57,"h":1.57,"cmd":0,"megs":"hello!"}

ESP_NOW_SINGLE (T=306)           Send control to one ESP-NOW device.
Example: {"T":306,"mac":"FF:FF:FF:FF:FF:FF","dev":0,"b":0,"s":0,"e":1.57,"h":1.57,"cmd":0,"megs":"hello!"}

WIFI_ON_BOOT (T=401)             Configure WiFi startup mode.
Example: {"T":401,"cmd":3}

SET_AP (T=402)                   Set WiFi access-point credentials.
Example: {"T":402,"ssid":"UGV","password":"12345678"}

SET_STA (T=403)                  Set WiFi station credentials.
Example: {"T":403,"ssid":"na","password":"ps"}

WIFI_APSTA (T=404)               Configure AP+STA WiFi.
Example: {"T":404,"ap_ssid":"UGV","ap_password":"12345678","sta_ssid":"na","sta_password":"ps"}

WIFI_INFO (T=405)                Get WiFi information.
Example: {"T":405}

WIFI_CONFIG_CREATE_BY_STATUS (T=406) Create WiFi config from current status.
Example: {"T":406}

WIFI_CONFIG_CREATE_BY_INPUT (T=407) Create WiFi config from input.
Example: {"T":407,"mode":3,"ap_ssid":"UGV","ap_password":"12345678","sta_ssid":"na","sta_password":"ps"}

WIFI_STOP (T=408)                Stop WiFi.
Example: {"T":408}

SET_SERVO_ID (T=501)             Change a servo ID.
Example: {"T":501,"raw":1,"new":11}

SET_MIDDLE (T=502)              Set a servo middle position.
Example: {"T":502,"id":11}

SET_SERVO_PID (T=503)            Set a servo P value.
Example: {"T":503,"id":14,"p":16}

REBOOT (T=600)                   Reboot the controller.
Example: {"T":600}

FREE_FLASH_SPACE (T=601)         Get free flash space.
Example: {"T":601}

BOOT_MISSION_INFO (T=602)        Get boot mission information.
Example: {"T":602}

RESET_BOOT_MISSION (T=603)       Reset the boot mission.
Example: {"T":603}

NVS_CLEAR (T=604)                Clear NVS configuration.
Example: {"T":604}

INFO_PRINT (T=605)               Set serial information-print mode.
Example: {"T":605,"cmd":1}

MM_TYPE_SET (T=900)              Set main/module type.
Example: {"T":900,"main":1,"module":0}</pre>
    </section>
</main>

<script>
let streaming = false;
let streamTimer = null;
let streamBusy = false;
let streamSamples = 0;

function jsonSend() {
    const input = document.getElementById("jsonData");
    const response = document.getElementById("response");
    let value = input.value.trim();

    if (!value) {
        response.textContent = "Enter a JSON command first.";
        return;
    }

    try {
        JSON.parse(value);
    } catch (error) {
        response.textContent = "Invalid JSON: " + error.message;
        return;
    }

    const parsed = JSON.parse(value);
    const isLatencyTest = parsed.T === 998;
    const start = performance.now();

    if (isLatencyTest) {
        parsed.echo = Date.now();
        value = JSON.stringify(parsed);
    }

    const xhr = new XMLHttpRequest();
    xhr.onreadystatechange = function() {
        if (this.readyState === 4) {
            if (this.status !== 200) {
                response.textContent = "HTTP error " + this.status;
                return;
            }

            if (isLatencyTest) {
                const rtt = performance.now() - start;
                response.textContent =
                    "Latency test\n" +
                    "Round-trip time: " + rtt.toFixed(2) + " ms\n" +
                    "Approx. one-way: " + (rtt / 2).toFixed(2) + " ms";
            } else {
                response.textContent = this.responseText;
            }
        }
    };
    xhr.open("GET", "js?json=" + encodeURIComponent(value), true);
    xhr.send();
}

function toggleStreaming() {
    streaming = document.getElementById("streamToggle").checked;

    if (streaming) {
        streamSamples = 0;
        document.getElementById("streamStatus").textContent = "Streaming enabled";
        streamTelemetry();
        streamTimer = setInterval(streamTelemetry, 100);
    } else {
        if (streamTimer !== null) {
            clearInterval(streamTimer);
            streamTimer = null;
        }
        document.getElementById("streamStatus").textContent =
            "Streaming disabled" + (streamSamples ? " (" + streamSamples + " samples)" : "");
    }
}

function streamTelemetry() {
    if (!streaming || streamBusy) return;

    streamBusy = true;

    const xhr = new XMLHttpRequest();
    xhr.onreadystatechange = function() {
        if (this.readyState === 4) {
            streamBusy = false;

            if (this.status === 200) {
                streamSamples++;
                document.getElementById("streamStatus").textContent =
                    "Streaming enabled (" + streamSamples + " samples)";
            }
        }
    };

    xhr.onerror = function() {
        streamBusy = false;
    };

    xhr.open("GET", "js?json=" + encodeURIComponent(JSON.stringify({"T":147})), true);
    xhr.send();
}

document.getElementById("jsonData").addEventListener("keydown", function(event) {
    if (event.key === "Enter") {
        jsonSend();
    }
});
</script>
</body>
</html>
)rawliteral";

#endif
