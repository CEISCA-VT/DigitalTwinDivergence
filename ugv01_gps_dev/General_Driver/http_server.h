#include "web_page.h"

// Create AsyncWebServer object on port 80
WebServer server(80);

void handleRoot(){
  server.send(
    200,
    "text/html",
    "<!doctype html><html><head><title>UGV01 Telemetry</title>"
    "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
    "</head><body><h1>UGV01 Telemetry Firmware</h1>"
    "<p>This experiment firmware exposes combined telemetry at "
    "<code>/telemetry</code>.</p>"
    "<p>Use the Python digital-twin dashboard for live contract evaluation.</p>"
    "</body></html>"
  );
}

void webCtrlServer(){
  server.on("/", handleRoot);

  server.on("/telemetry", [](){
    getAllTelemetryData();
    serializeJson(jsonInfoHttp, jsonFeedbackWeb);
    server.send(200, "application/json", jsonFeedbackWeb);
    jsonFeedbackWeb = "";
    jsonInfoHttp.clear();
    jsonCmdReceive.clear();
  });

  server.on("/js", [](){
    String jsonCmdWebString = server.arg(0);
    deserializeJson(jsonCmdReceive, jsonCmdWebString);
    jsonCmdReceiveHandler();
    serializeJson(jsonInfoHttp, jsonFeedbackWeb);
    server.send(200, "application/json", jsonFeedbackWeb);
    jsonFeedbackWeb = "";
    jsonInfoHttp.clear();
    jsonCmdReceive.clear();
  });

  // Start server
  server.begin();
  Serial.println("Server Starts.");
}

void initHttpWebServer(){
  webCtrlServer();
}
