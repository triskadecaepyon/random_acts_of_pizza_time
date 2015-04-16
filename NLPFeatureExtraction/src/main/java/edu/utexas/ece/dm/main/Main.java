package edu.utexas.ece.dm.main;

import java.io.IOException;

import com.fasterxml.jackson.core.JsonParseException;
import com.fasterxml.jackson.databind.JsonMappingException;

import edu.utexas.ece.dm.lib.DataProcessor;

public class Main {
	
	private static final String TRAINING_DATA_FILE = "./src/test/resources/train.json";
	private static final String TESTING_DATA_FILE = "./src/test/resources/test.json";
	private static final String TRAINING_RESULT_FILE = "./src/test/resources/train.csv";
	private static final String TESTING_RESULT_FILE = "./src/test/resources/test.csv";
	
	public static void main(String[] args) throws JsonParseException, JsonMappingException, IOException {
		// proc training data
		DataProcessor dp_train = new DataProcessor(TRAINING_DATA_FILE, TRAINING_RESULT_FILE);
		dp_train.process();
		
		// proc testing data
		DataProcessor dp_test = new DataProcessor(TESTING_DATA_FILE, TESTING_RESULT_FILE);
		dp_test.process();
	}
}
