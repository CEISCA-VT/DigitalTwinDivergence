#include "web_page.h"

// Create AsyncWebServer object on port 80
WebServer server(80);

void handleRoot(){
  server.send(200, "text/html", index_html); //Send web page
}

void webCtrlServer(){
  server.on("/", handleRoot);

  server.on("/js", [](){
    String jsonCmdWebString = server.arg("json");
    if (jsonCmdWebString.length() == 0) jsonCmdWebString = server.arg(0);
    deserializeJson(jsonCmdReceive, jsonCmdWebString);

    // T=998 is a network latency test. The browser measures the HTTP RTT.
    if (jsonCmdReceive["T"].as<int>() == 998) {
      jsonInfoHttp.clear();
      jsonInfoHttp["T"] = 998;
      jsonInfoHttp["echo"] = jsonCmdReceive["echo"] | 0;
      jsonInfoHttp["ugv_ms"] = millis();
    } else {
      jsonCmdReceiveHandler();
    }
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