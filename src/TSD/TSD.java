package TSD;

import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.util.*;

import weka.core.Instances;

public class TSD extends Instances {

	/**
	 * The true class labels of data set.
	 */
	int[] classLabels;
	/**
	 * k clusters(K-Means).
	 */
	int k;
	/**
	 * n clusters(CFSFDP).
	 */
	int n;
	/**
	 * cluster centers (K-Means).
	 */
	double[][] centerskMeans;
	/**
	 * predicted labels using K-Means.
	 */
	int[] predictedLablesKmeans;
	/**
	 * The size of each block
	 */
	int[] blockSizes;
	/**
	 * The block Information table.
	 */
	int[][] blockInformation;
	/**
	 * Select the representative points.
	 */
	double[][] representativePoints;
	/**
	 * rho.
	 */
	double[] rho;
	/**
	 * ordrho.
	 */
	int[] ordrho;

	/**
	 * delta.
	 */
	double[] delta;
	/**
	 * master.
	 */
	int[] master;
	/**
	 * priority.
	 */
	double[] priority;
	/**
	 * priority index.
	 */
	int[] ordpriority;

	/**
	 * maximalDistance.
	 */
	double maximalDistance;
	/**
	 * cluster centers (CFSFDP).
	 */
	int[] centersDensity;
	/**
	 * predicted labels.
	 */
	int[] predictedLabels;
	/**
	 * clusterIndices.
	 */
	int[] clusterIndices;
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
	public TSD(Reader paraReader) throws IOException, Exception {
		super(paraReader);

		classLabels = new int[numInstances()];
		for (int i = 0; i < numInstances(); i++) {
			classLabels[i] = (int) instance(i).value(numAttributes() - 1);

		}// Of for i

	}// Of the first constructor

	/**
	 ********************************** 
	 * Stage I. 2-round K-means clustering.
	 ********************************** 
	 */

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
		predictedLablesKmeans = new int[numInstances()];
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
	 * Step 5. 2-round K-Means clustering.
	 **************** 
	 */
	public void clusterKMeans(int paraK) {
		// Initialize
		k = paraK;
		centerskMeans = new double[k][numAttributes() - 1];
		int tempNumber = 0;

		// Select centers
		randomSelectCenters();

		// Cluster and mean
		while (tempNumber < 2) {
			// Cluster

			clusterUsingCenters();

			// Mean
			meanAsCenters();

			representativePoints = centerskMeans;

			tempNumber++;

			
		}// Of while
	}// Of cluster

	/**
	 ********************************** 
	 * Stage II. density clustering.
	 ********************************** 
	 */

	/**
	 ********************************** 
	 * Step 1. Compute block information according to the kMeans cluster.
	 ********************************** 
	 */
	public void computeBlockInformation() {
		int tempBlocks = centerskMeans.length;
		blockInformation = new int[centerskMeans.length][];

		for (int i = 0; i < tempBlocks; i++) {
			// Scan to see how many elements
			int tempElements = 0;
			for (int j = 0; j < numInstances(); j++) {
				if (predictedLablesKmeans[j] == i) {
					tempElements++;
				}// Of if
			}// Of for k

			// Copy to the list
			blockInformation[i] = new int[tempElements];
			tempElements = 0;
			for (int j = 0; j < numInstances(); j++) {
				if (predictedLablesKmeans[j] == i) {
					blockInformation[i][tempElements] = j;
					tempElements++;
				}// Of if
			}// Of for k
		}// Of for i
	}// Of computeBlockInformation

	/**
	 ********************************** 
	 * Step 2. Compute rho
	 ********************************** 
	 */
	public void computeRho() {
		rho = new double[k];

		for (int i = 0; i < blockInformation.length; i++) {
			int tempElements = 0;
			for (int j = 0; j < blockInformation[i].length; j++) {
				tempElements++;
				rho[i] = tempElements;
			}// of for j

		}// of for i

	}// Of computeRho

	/**
	 ********************************** 
	 * Step 3. Compute delta
	 ********************************** 
	 */
	public void computeDelta() {
		delta = new double[k];
		master = new int[k];
		ordrho = new int[k];
		ordrho = mergeSortToIndices(rho);
		
		computeMaximalDistance(representativePoints);
		delta[ordrho[0]] = maximalDistance;

		for (int i = 1; i < k; i++) {
			delta[ordrho[i]] = Double.MAX_VALUE;
			for (int j = 0; j <= i - 1; j++) {
				if (distanceArray(representativePoints[ordrho[i]],
						representativePoints[ordrho[j]]) < delta[ordrho[i]]) {
					delta[ordrho[i]] = distanceArray(
							representativePoints[ordrho[i]],
							representativePoints[ordrho[j]]);
					master[ordrho[i]] = ordrho[j];
				} // of if
			}// of for j
		}// of for i

	}// Of computeDelta

	/**
	 ********************************** 
	 * Step 3. Compute priority.
	 ********************************** 
	 */
	public void computePriority() {
		priority = new double[k];
		for (int i = 0; i < k; i++) {
			priority[i] = rho[i] * delta[i];
		}// Of for i
	}// Of computePriority

	/**
	 ********************************** 
	 * Step 4. Compute centers
	 ********************************** 
	 */
	public void computeCentersDensity(int paraK) {
		n = paraK;
		centersDensity = new int[n];
		ordpriority = new int[k];

		computePriority();

		ordpriority = mergeSortToIndices(priority);

		for (int i = 0; i < n; i++) {
			centersDensity[i] = ordpriority[i];
		}// of for i

	}// Of computeCenters

	/**
	 ********************************** 
	 * Step 5. Cluster according to the centers
	 ********************************** 
	 */
	public void clusterDensity() {

		predictedLabels = new int[numInstances()];
		int[] cl = new int[k];
		clusterIndices = new int[k];

		for (int i = 0; i < k; i++) {
			cl[i] = -1;
		}// of for i

		for (int i = 0; i < n; i++) {

			cl[centersDensity[i]] = i;

		}// of for i

		for (int i = 0; i < k; i++) {

			if (cl[ordrho[i]] == -1) {

				cl[ordrho[i]] = cl[master[ordrho[i]]];

			}// of if
		}// of for i

		for (int i = 0; i < k; i++) {
			clusterIndices[i] = centersDensity[cl[i]];
		}

		for (int i = 0; i < blockInformation.length; i++) {

			for (int j = 0; j < blockInformation[i].length; j++) {

				predictedLabels[blockInformation[i][j]] = cl[i];

			}// of if
		}// of for i

	}// Of clusterWithCenters

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
		int[][] distributionMatrix = new int[maximal(predictedLabels) + 1][maximal(classLabels) + 1];

		// Fill the matrix
		for (int i = 0; i < predictedLabels.length; i++) {
			distributionMatrix[predictedLabels[i]][classLabels[i]]++;
		}// Of for i

		double tempPurity = 0;
		for (int i = 0; i < distributionMatrix.length; i++) {
			tempPurity += maximal(distributionMatrix[i]);
		}// Of for i

		purity = tempPurity / numInstances();

	}// Of computPurity

	public void computExternalMeasures() {
		ss = sd = ds = dd = 0;
		for (int i = 0; i < predictedLabels.length - 1; i++) {
			for (int j = i + 1; j < predictedLabels.length; j++) {
				if ((predictedLabels[i] == predictedLabels[j])
						&& (classLabels[i] == classLabels[j])) {
					ss++;
				} else if ((predictedLabels[i] == predictedLabels[j])
						&& (classLabels[i] != classLabels[j])) {
					sd++;
				} else if ((predictedLabels[i] != predictedLabels[j])
						&& (classLabels[i] == classLabels[j])) {
					ds++;
				} else {
					dd++;
				}// Of if
			}// Of for j
		}// Of for i
		jc = (ss + 0.0) / (ss + sd + ds);
		fmi = (ss + 0.0) / Math.sqrt((ss + ds) * (ss + sd));
		ri = 2.0 * (ss + dd)
				/ (predictedLabels.length * (predictedLabels.length - 1));
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
	 *************** 
	 * Compute the distance.
	 **************** 
	 */
	public double distanceArray(double[] paraArrayFirst,
			double[] paraArraySecond) {

		double tempDistance = 0;
		for (int i = 0; i < numAttributes() - 1; i++) {
			tempDistance += Math.abs(paraArrayFirst[i] - paraArraySecond[i]);
		}// Of for i

		return tempDistance;
	}// Of distance

	/**
	 ********************************** 
	 * Merge sort in descendant order to obtain an index array. The original
	 * array is unchanged.<br>
	 * Examples: input [1.2, 2.3, 0.4, 0.5], output [1, 0, 3, 2].<br>
	 * input [3.1, 5.2, 6.3, 2.1, 4.4], output [2, 1, 4, 0, 3].
	 * 
	 * @author Fan Min 2016/09/09
	 * 
	 * @param paraArray
	 *            the original array
	 * @return The sorted indices.
	 ********************************** 
	 */

	public static int[] mergeSortToIndices(double[] paraArray) {
		int tempLength = paraArray.length;
		int[][] resultMatrix = new int[2][tempLength];// 两个维度交换存储排序tempIndex控制

		// Initialize
		int tempIndex = 0;
		for (int i = 0; i < tempLength; i++) {
			resultMatrix[tempIndex][i] = i;
		} // Of for i
			// Merge
		int tempCurrentLength = 1;
		// The indices for current merged groups.
		int tempFirstStart, tempSecondStart, tempSecondEnd;
		while (tempCurrentLength < tempLength) {

			// Divide into a number of groups
			// Here the boundary is adaptive to array length not equal to 2^k.

			for (int i = 0; i < Math.ceil(tempLength + 0.0 / tempCurrentLength) / 2; i++) {
				// Boundaries of the group

				tempFirstStart = i * tempCurrentLength * 2;

				tempSecondStart = tempFirstStart + tempCurrentLength;

				tempSecondEnd = tempSecondStart + tempCurrentLength - 1;
				if (tempSecondEnd >= tempLength) {
					tempSecondEnd = tempLength - 1;
				} // Of if

				// Merge this group
				int tempFirstIndex = tempFirstStart;
				int tempSecondIndex = tempSecondStart;
				int tempCurrentIndex = tempFirstStart;

				if (tempSecondStart >= tempLength) {
					for (int j = tempFirstIndex; j < tempLength; j++) {
						resultMatrix[(tempIndex + 1) % 2][tempCurrentIndex] = resultMatrix[tempIndex % 2][j];
						tempFirstIndex++;
						tempCurrentIndex++;
					} // Of for j
					break;
				} // Of if

				while ((tempFirstIndex <= tempSecondStart - 1)
						&& (tempSecondIndex <= tempSecondEnd)) {

					if (paraArray[resultMatrix[tempIndex % 2][tempFirstIndex]] >= paraArray[resultMatrix[tempIndex % 2][tempSecondIndex]]) {

						resultMatrix[(tempIndex + 1) % 2][tempCurrentIndex] = resultMatrix[tempIndex % 2][tempFirstIndex];
						int a = (tempIndex + 1) % 2;

						tempFirstIndex++;
					} else {
						resultMatrix[(tempIndex + 1) % 2][tempCurrentIndex] = resultMatrix[tempIndex % 2][tempSecondIndex];
						int b = (tempIndex + 1) % 2;

						tempSecondIndex++;
					} // Of if
					tempCurrentIndex++;

				} // Of while

				// Remaining part

				for (int j = tempFirstIndex; j < tempSecondStart; j++) {
					resultMatrix[(tempIndex + 1) % 2][tempCurrentIndex] = resultMatrix[tempIndex % 2][j];

					tempCurrentIndex++;

				} // Of for j
				for (int j = tempSecondIndex; j <= tempSecondEnd; j++) {
					resultMatrix[(tempIndex + 1) % 2][tempCurrentIndex] = resultMatrix[tempIndex % 2][j];

					tempCurrentIndex++;
				} // Of for j

			} // Of for i

			tempCurrentLength *= 2;
			tempIndex++;
		} // Of while

		return resultMatrix[tempIndex % 2];
	}// Of mergeSortToIndices

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
	 ********************************** 
	 * Compute the maximal distance.
	 ********************************** 
	 */
	public double computeMaximalDistance(double[][] paraArray) {
		maximalDistance = 0;
		double tempDistance;
		for (int i = 0; i < paraArray.length; i++) {
			for (int j = 0; j < paraArray[i].length; j++) {
				tempDistance = distanceArray(paraArray[i], paraArray[j]);
				if (maximalDistance < tempDistance) {
					maximalDistance = tempDistance;
				}// Of if
			}// Of for j
		}// Of for i

		return maximalDistance;
	}// Of setDistanceMeasure

	/**
	 ******************* 
	 * TSD test.
	 ******************* 
	 */
	public static void TSDTest() {
		String arffFilename = "E:/data/iris.arff";

		try {
			long startread = System.currentTimeMillis();
			FileReader fileReader = new FileReader(arffFilename);

			TSD tempData = new TSD(fileReader);
			fileReader.close();

			long endread = System.currentTimeMillis();
			System.out.println("读取文件花费时间" + (endread - startread) + "毫秒!");
			long startkmeans = System.currentTimeMillis();
			tempData.clusterKMeans(12);
			long endkmeans = System.currentTimeMillis();
			System.out.println("计算kmeans花费时间" + (endkmeans - startkmeans) + "毫秒!");
			//System.out.println("centerkmeans"
			//		+ Arrays.deepToString(tempData.centerskMeans));
			long startdp = System.currentTimeMillis();
			tempData.computeBlockInformation();
			//System.out.println("blockInformation"
			//		+ Arrays.deepToString(tempData.blockInformation));
			//System.out.println("representativePoints"
			//		+ Arrays.deepToString(tempData.representativePoints));
			tempData.computeRho();
			//System.out.println("rho" + Arrays.toString(tempData.rho));

			tempData.computeDelta();
			//System.out.println("delta" + Arrays.toString(tempData.delta));
			//System.out.println("master" + Arrays.toString(tempData.master));
			tempData.computeMaximalDistance(tempData.representativePoints);
			//System.out.println("maximial" + tempData.maximalDistance);
			tempData.computeCentersDensity(3);
			//System.out.println("centersDensity"
			//		+ Arrays.toString(tempData.centersDensity));
			tempData.clusterDensity();
			//System.out.println("predictedLabels"
			//		+ Arrays.toString(tempData.predictedLabels));
			long enddp = System.currentTimeMillis();
			System.out.println("计算dp花费时间" + (enddp - startdp) + "毫秒!");
			tempData.computPurity();
			System.out.println("The purity is" + tempData.purity);
			tempData.computExternalMeasures();
			System.out.println("The jc is" + tempData.jc);
			System.out.println("The fmi is" + tempData.fmi);
			System.out.println("The ri is" + tempData.ri);

		} catch (Exception ee) {
			System.out.println("Error occurred while trying to read \'"
					+ arffFilename + "\' in TSDTest().\r\n" + ee);
		}// Of try
	}// Of densityTest

	public static void main(String[] args) {

		long start = System.currentTimeMillis();
		TSDTest();

		long end = System.currentTimeMillis();
		System.out.println("总花费时间" + (end - start) + "毫秒!");

	}// Of main

}// Of TSD

