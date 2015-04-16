package edu.utexas.ece.dm.entity;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

@JsonIgnoreProperties(ignoreUnknown = true)
public class Data {
	
	@JsonProperty("request_text")
	private String requestText;
	@JsonProperty("request_text_edit_aware")
	private String requestTextEditAware;
	@JsonProperty("request_title")
	private String requestTitle;
	@JsonProperty("requester_username")
	private String requesterUserName;
	@JsonProperty("request_id")
	private String requestId;
	@JsonProperty("requester_received_pizza")
	private boolean requesterReceivedPizza;
	
	public String getRequestText() {
		return requestText;
	}
	public void setRequestText(String requestText) {
		this.requestText = requestText;
	}
	public String getRequestTextEditAware() {
		return requestTextEditAware;
	}
	public void setRequestTextEditAware(String requestTextEditAware) {
		this.requestTextEditAware = requestTextEditAware;
	}
	public String getRequestTitle() {
		return requestTitle;
	}
	public void setRequestTitle(String requestTitle) {
		this.requestTitle = requestTitle;
	}
	public String getRequesterUserName() {
		return requesterUserName;
	}
	public void setRequesterUserName(String requesterUserName) {
		this.requesterUserName = requesterUserName;
	}
	public String getRequestId() {
		return requestId;
	}
	public void setRequestId(String requestId) {
		this.requestId = requestId;
	}
	public boolean isRequesterReceivedPizza() {
		return requesterReceivedPizza;
	}
	public void setRequesterReceivedPizza(boolean requesterReceivedPizza) {
		this.requesterReceivedPizza = requesterReceivedPizza;
	}
}
