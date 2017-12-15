package TSD;

import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.util.Arrays;

import weka.core.Instances;

public class CFDP extends Instances {
	/**
	 * The true class labels of data set.
	 */
	int[] classLabels;
	/**
	 * rho.
	 */
	double[] rho;
	/**
	 * dc.
	 */
	double dc;

	double maximalDistance;
	/**
	 * rho排序后的数组索引数组.
	 */
	int[] ordrho;

	/**
	 * 实例的最小距离.
	 */
	double[] delta;

	/**
	 * 实例的master.
	 */
	int[] master;
	/**
	 * 实例的优先级.
	 */
	double[] priority;

	/**
	 * 聚类中心
	 */
	int[] centers;

	/**
	 * 簇信息，我属于哪一簇？
	 */
	int[] cl;
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
	public CFDP(Reader paraReader) throws IOException, Exception {
		super(paraReader);

		classLabels = new int[numInstances()];
		for (int i = 0; i < numInstances(); i++) {
			classLabels[i] = (int) instance(i).value(numAttributes() - 1);

		}// Of for i

	}// Of the first constructor

	/**
	 ********************************** 
	 * Step 1. compute distance
	 ********************************** 
	 */

	/**
	 ********************************** 
	 * Manhattan distance.
	 ********************************** 
	 */
	public double manhattan(int paraI, int paraJ) {
		double tempDistance = 0;

		for (int i = 0; i < numAttributes() - 1; i++) {
			tempDistance += Math.abs(instance(paraI).value(i)
					- instance(paraJ).value(i));
		}// of for i

		return tempDistance;
	}// of manhattan

	/**
	 ********************************** 
	 * Step 2. Compute rho
	 ********************************** 
	 */

	public void computeRho() {
		rho = new double[numInstances()];

		for (int i = 0; i < numInstances() - 1; i++) {
			for (int j = i + 1; j < numInstances(); j++) {
				if (manhattan(i, j) < dc) {
					rho[i]++;
					rho[j]++;
				}// of if
			}// of for j
		}// of for i

	}// of computeRho

	/**
	 ********************************** 
	 * Set dc.
	 ********************************** 
	 */
	public void setDc(double paraPercentage) {

		dc = maximalDistance * paraPercentage;
	}// of setDc

	/**
	 ********************************** 
	 * Compute delta
	 ********************************** 
	 */

	public void computeDelta() {
		delta = new double[numInstances()];
		master = new int[numInstances()];
		ordrho = new int[numInstances()];

		// Step 1. rho排序
		ordrho = mergeSortToIndices(rho);

		// Step 2. delta[ordrho[0]]
		delta[ordrho[0]] = maximalDistance;

		// Step 3. 找最小距离

		for (int i = 1; i < numInstances(); i++) {
			delta[ordrho[i]] = maximalDistance;
			for (int j = 0; j <= i - 1; j++) {
				if (manhattan(ordrho[i], ordrho[j]) < delta[ordrho[i]]) {
					delta[ordrho[i]] = manhattan(ordrho[i], ordrho[j]);
					master[ordrho[i]] = ordrho[j];
				}// of if
			}// of for j
		}// of for i

	}// of computeDelta

	/**
	 ********************************** 
	 * Compute priority.
	 ********************************** 
	 */
	public void computePriority() {
		priority = new double[numInstances()];

		for (int i = 0; i < numInstances(); i++) {

			priority[i] = rho[i] * delta[i];

		}// of for i

	}// of computePriority

	// Step 2. 根据priority排序，从上到下依次选择k个中心

	public void computeCenters(int paraNumbers) {
		centers = new int[paraNumbers];

		// Step 1. 对priority排序
		int[] ordPriority = mergeSortToIndices(priority);

		// Step 2. 选择k个中心
		for (int i = 0; i < paraNumbers; i++) {
			centers[i] = ordPriority[i];
		}// of for i

	}// of computeCenters

	/**
	 ********************************** 
	 * 1.5. 根据中心点进行聚类
	 ********************************** 
	 */
	public void clusterWithCenters() {

		// Step 1. 初始化
		cl = new int[numInstances()];

		for (int i = 0; i < numInstances(); i++) {
			cl[i] = -1;
		}// of for i

		// Step 2. 给中心点分配类标签

		int tempNumber = 0;
		int tempCluster = 0;
		for (int i = 0; i < numInstances(); i++) {
			if (tempNumber < centers.length) {
				cl[centers[i]] = tempCluster;
				tempNumber++;
				tempCluster++;
			}// of if
		}// of for i

		/*
		 * // Step 2 中心给标记 // System.out.println("this is the test 1" ); for
		 * (int i = 0; i < centers.length; i++) { cl[centers[i]] = i; }// of for
		 * i
		 */

		// Step 3. 给其余点分类类标签（类标签与其master一致）

		for (int i = 0; i < numInstances(); i++) {
			if (cl[ordrho[i]] == -1) {
				cl[ordrho[i]] = cl[master[ordrho[i]]];
			}// of if
		}// of for i

	}// of clusterWithCenters

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
		int[][] distributionMatrix = new int[maximal(cl) + 1][maximal(classLabels) + 1];

		// Fill the matrix
		for (int i = 0; i < cl.length; i++) {
			distributionMatrix[cl[i]][classLabels[i]]++;
		}// Of for i

		double tempPurity = 0;
		for (int i = 0; i < distributionMatrix.length; i++) {
			tempPurity += maximal(distributionMatrix[i]);
		}// Of for i

		purity = tempPurity / numInstances();

	}// Of computPurity

	public void computExternalMeasures() {
		ss = sd = ds = dd = 0;
		for (int i = 0; i < cl.length - 1; i++) {
			for (int j = i + 1; j < cl.length; j++) {
				if ((cl[i] == cl[j]) && (classLabels[i] == classLabels[j])) {
					ss++;
				} else if ((cl[i] == cl[j])
						&& (classLabels[i] != classLabels[j])) {
					sd++;
				} else if ((cl[i] != cl[j])
						&& (classLabels[i] == classLabels[j])) {
					ds++;
				} else {
					dd++;
				}// Of if
			}// Of for j
		}// Of for i
		jc = (ss + 0.0) / (ss + sd + ds);
		fmi = (ss + 0.0) / Math.sqrt((ss + ds) * (ss + sd));
		ri = 2.0 * (ss + dd) / (cl.length * (cl.length - 1));
	}// Of computExternalMeasures

	/**
	 ********************************** 
	 * Other function.
	 ********************************** 
	 */

	/**
	 ********************************** 
	 * Compute maximal distance.
	 ********************************** 
	 */

	public void computeMaximalDistance() {
		// maximalDistance = 0;
		double tempDistance;
		for (int i = 0; i < numInstances(); i++) {
			for (int j = 0; j < numInstances(); j++) {
				tempDistance = manhattan(i, j);
				if (maximalDistance < tempDistance) {
					maximalDistance = tempDistance;
				}// of if
			}// of for j
		}// of for i

	}// of computeMaximalDistance

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
	 ******************* 
	 * 测试函数
	 ******************* 
	 */

	public static void Test() {

		String arffFilename = "E:/data/jain.arff";

		try {
			long startread = System.currentTimeMillis();
			FileReader fileReader = new FileReader(arffFilename);
			CFDP tempData = new CFDP(fileReader);
			fileReader.close();
			long endread = System.currentTimeMillis();
			System.out.println("读取文件花费时间" + (endread - startread) + "毫秒!");
			tempData.setClassIndex(tempData.numAttributes() - 1);
			// System.out.println(tempData);

			// tempDistance = tempData.manhattan(1, 2);
			// System.out.println("tempDistance" + tempDistance);
			long startDensity = System.currentTimeMillis();
			tempData.computeMaximalDistance();
			tempData.setDc(1);
			// System.out.println("dc" + tempData.dc);
			tempData.computeRho();
			// System.out.println("rho" + Arrays.toString(tempData.rho));
			tempData.computeDelta();
			// System.out.println("delta" + Arrays.toString(tempData.delta));
			// System.out.println("master" + Arrays.toString(tempData.master));
			tempData.computePriority();
			tempData.computeCenters(2);
			// System.out.println("centers" +
			// Arrays.toString(tempData.centers));

			tempData.clusterWithCenters();
			long endDensity = System.currentTimeMillis();
			System.out
					.println("计算dp花费时间" + (endDensity - startDensity) + "毫秒!");
			// System.out.println("cl"
			// + Arrays.toString(tempData.cl));
			tempData.computPurity();
			System.out.println("The purity is" + tempData.purity);
			tempData.computExternalMeasures();
			System.out.println("The jc is" + tempData.jc);
			System.out.println("The fmi is" + tempData.fmi);
			System.out.println("The ri is" + tempData.ri);

		} catch (Exception ee) {
		}// of try

	}// of densityTest

	/**
	 ********************************** 
	 * Main.
	 ********************************** 
	 */
	public static void main(String args[]) {

		Test();
	}// of main

}// of Density
