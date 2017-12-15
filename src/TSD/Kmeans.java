package TSD;

import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.util.*;

import weka.core.Instances;

public class Kmeans extends Instances {

	/**
	 * The true class labels of data set.
	 */
	int[] classLabels;
	/**
	 * cluster centers (K-Means).
	 */
	double[][] centerskMeans;
	/**
	 * k clusters(K-Means).
	 */
	int k;
	/**
	 * predicted labels using K-Means.
	 */
	int[] predictedLablesKmeans;
	/**
	 * The size of each block
	 */
	int[] blockSizes;
	/**
	 * Is the cluster information changed?
	 */
	boolean clusterChanged;
	/**
	 * purity.
	 */
	double purity;

	double ss;

	double sd;

	double ds;

	double dd;

	double jc;

	double fmi;

	double ri;

	double nmi;

	/**
	 ********************************** 
	 * Read from a reader
	 ********************************** 
	 */
	public Kmeans(Reader paraReader) throws IOException, Exception {
		super(paraReader);

		classLabels = new int[numInstances()];
		for (int i = 0; i < numInstances(); i++) {
			classLabels[i] = (int) instance(i).value(numAttributes() - 1);

		}// Of for i

	}// Of the first constructor

	/**
	 *************** 
	 * Step 1. Select the centers randomly.
	 **************** 
	 */

	public void randomSelectCenters() {

		int[] tempIndex = generateRandomSequence(numInstances());

		for (int i = 0; i < k; i++) {
			for (int j = 0; j < numAttributes() - 1; j++) {
				centerskMeans[i][j] = instance(tempIndex[i]).value(j);
			}// of for j

		}// of for i
	}// of randomSelcetCenters

	/**
	 *************** 
	 * Step 2. Compute the distance.
	 **************** 
	 */

	public double computeDistance(int paraI, double[] paraArray) {

		double tempDistance = 0;
		for (int i = 0; i < numAttributes() - 1; i++) {
			tempDistance += Math.abs(instance(paraI).value(i) - paraArray[i]);
		}// Of for i

		return tempDistance;
	}// Of distance

	/**
	 *************** 
	 * Step 3. Cluster using centers.
	 **************** 
	 */

	public void clusterUsingCenters() {

		for (int i = 0; i < numInstances(); i++) {
			int tempIndex = 0;
			double tempDistance = Double.MAX_VALUE;

			for (int j = 0; j < centerskMeans.length; j++) {
				if (computeDistance(i, centerskMeans[j]) < tempDistance) {
					tempDistance = computeDistance(i, centerskMeans[j]);
					tempIndex = j;
				}// Of if
			}// Of for j

			if (predictedLablesKmeans[i] != tempIndex) {
				clusterChanged = true;
				predictedLablesKmeans[i] = tempIndex;
			}// Of if
		}// Of for i
	}// Of clusterUsingCenters

	/**
	 *************** 
	 * Step 4. Compute new centers using the mean value of each block.
	 **************** 
	 */
	public void meanAsCenters() {
		// Initialize
		blockSizes = new int[k];
		for (int i = 0; i < centerskMeans.length; i++) {
			blockSizes[i] = 0;
			for (int j = 0; j < centerskMeans[i].length; j++) {
				centerskMeans[i][j] = 0;
			}// Of for j
		}// Of for i

		// Scan all instances and sum
		for (int i = 0; i < numInstances(); i++) {
			blockSizes[predictedLablesKmeans[i]]++;
			for (int j = 0; j < numAttributes() - 1; j++) {
				centerskMeans[predictedLablesKmeans[i]][j] += instance(i)
						.value(j);
			}// Of for j
		}// Of for i

		// Divide
		for (int i = 0; i < centerskMeans.length; i++) {
			for (int j = 0; j < centerskMeans[i].length; j++) {
				centerskMeans[i][j] /= blockSizes[i];
			}// Of for j
		}// Of for i
	}// Of meanAsCenters

	/**
	 *************** 
	 * Step 5. Cluster.
	 **************** 
	 */
	public void cluster(int paraK) {
		// Initialize
		k = paraK;
		predictedLablesKmeans = new int[numInstances()];
		centerskMeans = new double[k][numAttributes() - 1];
		clusterChanged = true;

		// Select centers
		randomSelectCenters();

		// Cluster and mean
		while (true) {
			clusterChanged = false;

			// Cluster
			clusterUsingCenters();

			if (!clusterChanged) {
				break;
			}// Of if

			// Mean
			meanAsCenters();
		}// Of while
	}// Of cluster

	/**
	 *************** 
	 * Evaluation function.
	 **************** 
	 */

	/**
	 *************** 
	 * compute purity.
	 **************** 
	 */
	public void computPurity() {
		// Scan to determine the size of the distribution matrix
		purity = 0;
		int[][] distributionMatrix = new int[maximal(predictedLablesKmeans) + 1][maximal(classLabels) + 1];

		// Fill the matrix
		for (int i = 0; i < predictedLablesKmeans.length; i++) {
			distributionMatrix[predictedLablesKmeans[i]][classLabels[i]]++;
		}// Of for i

		double tempPurity = 0;
		for (int i = 0; i < distributionMatrix.length; i++) {
			tempPurity += maximal(distributionMatrix[i]);
		}// Of for i

		purity = tempPurity / numInstances();

	}// Of computPurity

	public void computExternalMeasures() {
		ss = sd = ds = dd = 0;
		for (int i = 0; i < predictedLablesKmeans.length - 1; i++) {
			for (int j = i + 1; j < predictedLablesKmeans.length; j++) {
				if ((predictedLablesKmeans[i] == predictedLablesKmeans[j])
						&& (classLabels[i] == classLabels[j])) {
					ss++;
				} else if ((predictedLablesKmeans[i] == predictedLablesKmeans[j])
						&& (classLabels[i] != classLabels[j])) {
					sd++;
				} else if ((predictedLablesKmeans[i] != predictedLablesKmeans[j])
						&& (classLabels[i] == classLabels[j])) {
					ds++;
				} else {
					dd++;
				}// Of if
			}// Of for j
		}// Of for i
		jc = (ss + 0.0) / (ss + sd + ds);
		fmi = (ss + 0.0) / Math.sqrt((ss + ds) * (ss + sd));
		ri = 2.0
				* (ss + dd)
				/ (predictedLablesKmeans.length * (predictedLablesKmeans.length - 1));
	}// Of computExternalMeasures

	/**
	 ********************************** 
	 * Other functions.
	 ********************************** 
	 */

	/**
	 ********************************** 
	 * Generate a random sequence of [0, n - 1].
	 * 
	 * @author Hengru Zhang, Revised by Fan Min 2013/12/24
	 * 
	 * @param paraLength
	 *            the length of the sequence
	 * @return an array of non-repeat random numbers in [0, paraLength - 1].
	 ********************************** 
	 */
	public static int[] generateRandomSequence(int paraLength) {
		Random random = new Random();
		// Initialize
		int[] tempResultArray = new int[paraLength];
		for (int i = 0; i < paraLength; i++) {
			tempResultArray[i] = i;
		}// Of for i

		// Swap some elements
		int tempFirstIndex, tempSecondIndex, tempValue;
		for (int i = 0; i < paraLength / 2; i++) {
			tempFirstIndex = random.nextInt(paraLength);
			tempSecondIndex = random.nextInt(paraLength);

			// Really swap elements in these two indices
			tempValue = tempResultArray[tempFirstIndex];
			tempResultArray[tempFirstIndex] = tempResultArray[tempSecondIndex];
			tempResultArray[tempSecondIndex] = tempValue;
		}// Of for i

		return tempResultArray;
	}// Of generateRandomSequence

	/**
	 ********************************** 
	 * Compute the maximal value.
	 ********************************** 
	 */

	public int maximal(int[] paraArray) {
		int tempMaximal = Integer.MIN_VALUE;
		for (int i = 0; i < paraArray.length; i++) {
			if (tempMaximal < paraArray[i]) {
				tempMaximal = paraArray[i];
			}// Of if
		}// Of for i

		return tempMaximal;
	}// Of maximal

	/**
	 ******************* 
	 * Kmeans test.
	 ******************* 
	 */
	public static void KmeansTest() {

		String arffFilename = "E:/data/iris.arff";

		try {
			long startread = System.currentTimeMillis();
			FileReader fileReader = new FileReader(arffFilename);

			Kmeans tempData = new Kmeans(fileReader);
			fileReader.close();
			long endread = System.currentTimeMillis();
			System.out.println("读取文件花费时间" + (endread - startread) + "毫秒!");
			long startkmeans = System.currentTimeMillis();
			// System.out.println(tempData);
			tempData.cluster(3);
			long endkmeans = System.currentTimeMillis();
			System.out.println("计算kmeans花费时间" + (endkmeans - startkmeans) + "毫秒!");
			// System.out.println("predictedLablesKmeans"
			//		+ Arrays.toString(tempData.predictedLablesKmeans));
			tempData.computPurity();
			System.out.println("The purity is" + tempData.purity);
			tempData.computExternalMeasures();
			System.out.println("The jc is" + tempData.jc);
			System.out.println("The fmi is" + tempData.fmi);
			System.out.println("The ri is" + tempData.ri);

		} catch (Exception ee) {
			System.out.println("Error occurred while trying to read \'"
					+ arffFilename + "\' in densityTest().\r\n" + ee);
		}// Of try
	}// Of KmeansTest

	public static void main(String[] args) {

		KmeansTest();

	}// Of main

}// Of Kmeans

