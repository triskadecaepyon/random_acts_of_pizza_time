package edu.utexas.ece.dm.lib;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import au.com.bytecode.opencsv.CSVWriter;

import com.fasterxml.jackson.core.JsonParseException;
import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.ObjectMapper;

import edu.utexas.ece.dm.entity.Data;
import edu.utexas.ece.dm.entity.Result;

public class DataProcessor {

	Data[] data;
	String file;
	
	public DataProcessor(String infile, String outfile) throws JsonParseException,
			JsonMappingException, IOException {
		ObjectMapper mapper = new ObjectMapper(); // can reuse, share globally
		data = mapper.readValue(new File(infile), Data[].class);
		this.file = outfile;
	}

	public Data[] getData() {
		return data;
	}

	public void process() throws JsonParseException, JsonMappingException, IOException {
		CSVWriter writer = new CSVWriter(new FileWriter(file), ',');
	     // feed in your array (or convert your data to an array)
	    String[] headers = "Title#Sentiment Score#isPizzaMentioned#Length#Truth".split("#");
	    // write header
	    writer.writeNext(headers);
	    
		for (Data itm : data) {
			Result result = new Result();
			int score = NLPProcessor.findSentiment(itm.getRequestTextEditAware());
			System.out.println(String.format("Title: %s, Score: %d, Received Pizza: %s", itm.getRequestTitle(), score, String.valueOf(itm.isRequesterReceivedPizza())));
			result.setTitle(itm.getRequestTitle());
			result.setTruth(itm.isRequesterReceivedPizza());
			result.setSentiment(score);
			result.setLength(itm.getRequestTextEditAware().length());
			result.setPizza(itm.getRequestTextEditAware().contains("pizza"));
			
			//write to file
			String temp = result.getTitle()+"#"+result.getSentiment()+"#"+result.isPizza()+"#"+result.getLength()+"#"+result.isTruth();
	    	String[] entry = temp.split("#");
	    	writer.writeNext(entry);
		}
		
		writer.close();
		
	}
}
