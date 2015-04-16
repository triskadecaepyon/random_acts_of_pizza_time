package edu.utexas.ece.dm.entity;

public class Result {
	
	private String title;
	private int sentiment;
	private boolean isPizza;
	private int length;
	private boolean truth;
	
	public int getSentiment() {
		return sentiment;
	}
	public void setSentiment(int sentiment) {
		this.sentiment = sentiment;
	}
	public int getLength() {
		return length;
	}
	public void setLength(int length) {
		this.length = length;
	}
	public String getTitle() {
		return title;
	}
	public void setTitle(String title) {
		this.title = title;
	}
	public boolean isPizza() {
		return isPizza;
	}
	public void setPizza(boolean isPizza) {
		this.isPizza = isPizza;
	}
	public boolean isTruth() {
		return truth;
	}
	public void setTruth(boolean truth) {
		this.truth = truth;
	}
	
}
