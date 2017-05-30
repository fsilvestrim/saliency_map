
package de.unibonn.informatik.ivs.set;


import java.io.*;
import java.util.*;
import java.awt.*;
import java.awt.image.*;

import ptolemy.plot.*;


/**
 * Main class to start the Saliency Evaluator Tool (SET).<br>
 * <br>
 * SET evaluates saliency maps with
 * defined ground truths of salient objects as described in <br>
 * <code>R. Achanta, S. Hemami, F. Estrada and S. Süsstrunk, Frequency-tuned Salient Region Detection, IEEE International Conference on Computer Vision and Pattern Recognition (CVPR), 2009.</code><br>
 * <a href="http://ivrg.epfl.ch/supplementary_material/RK_CVPR09/index.html">http://ivrg.epfl.ch/supplementary_material/RK_CVPR09/index.html</a><br>
 * <br>
 * Ground truths can be provided as either binary images or ground truth descriptions that define a set of rectangles.<br>
 * For further information on these descriptions see<br>
 * <code>Tie Liu, Jian Sun, Nan-Ning Zheng, Xiaoou Tang and Heung-Yeung Shum. Learning to Detect A Salient Object. In Proc. IEEE Cont. on Computer Vision and pattern Recognition (CVPR), Minneapolis, Minnesota, 2007.</code><br>
 * <a href="http://research.microsoft.com/en-us/um/people/jiansun/SalientObject/salient_object.htm">http://research.microsoft.com/en-us/um/people/jiansun/SalientObject/salient_object.htm</a><br>
 * <br>
 * Additionally SET provides two image copy operations:<br>
 * 1. Copy binary ground truth images whose salient objects are relatively smaller than a threshold<br>
 * 2. Copy images defined by image names existing in a specified directory<br>
 * <br>
 * Get information about the usage by starting SET with the <code>-h</code> parameter:<br>
 * <code>java -jar SaliencyEvaluationTool.jar -h</code>
 *
 *
 * @author		Bernd Wendt
 * @version		2011.0317
 *
 */
public class Main
{

	/**
	 * Constant indicating that application runs in evaluation mode.
	 * In evaluation mode a set of saliency maps is evaluated against a set of ground truth information.
	 */
	public static final int MODE_EVALUATE             = 0;

	/**
	 * Constant indicating that appliaction shall copy ground truth images whose ground truth object's relative
	 * size in image is smaller than a specified size.
	 */
	public static final int MODE_COPYSMALLGROUNDTRUTH = 1;

	/**
	 * Constant indicating that application shall copy images with names that are specified by images
	 * existing in a specified folder.
	 */
	public static final int MODE_COPYIMAGES           = 2;


	/**
	 * Current running mode.
	 */
	private static int mode = MODE_EVALUATE;

	/**
	 * Path where ground truth images can be found.
	 */
	private static String pathGroundTruthImages  = null;

	/**
	 * Path where a file specifying ground truths can be found.
	 */
	private static String pathGroundTruthFile    = null;

	/**
	 * Path where saliency map images can be found.
	 */
	private static String pathSaliencyMapImages  = null;

	/**
	 * Path where images shall be copied to.
	 */
	private static String pathImagesToCopy       = ".";

	/**
	 * Path where image files defining images to copy by thier name can be found.
	 */
	private static String pathDefiningImageNames = null;

	/**
	 * Path where results shall be written to.
	 */
	private static String pathResult             = ".";

	/**
	 * Determines the threshold used to generate binary ground truth maps from given
	 * greyvalue ground truth maps. Must be in [0.0; 1.0]
	 */
	private static double thresholdBinGroundTruth  = 0.5;

	/**
	 * Determines the maximum ground truth size to copy when in copy small ground truth mode.
	 */
	private static double thresholdSizeGroundTruth = 0.5;

	/**
	 * Determines if generated binary ground truth maps shall be saved as images to
	 * {@link #pathResult}.
	 */
	private static boolean saveGroundTruthImages = false;

	/**
	 * Holds relative ground truth sizes (to image size) by image name.
	 */
	private static Map<String, Double> relativeGtSizes;


	/**
	 * Main method to start from.
	 *
	 * @param args command line arguments
	 */
	public static void main(String[] args)
	{
		// determine what to do
		parseCommandLineParameters(args);

		// execute
		try
		{
			if (mode == MODE_COPYSMALLGROUNDTRUTH)
			{
				if (pathGroundTruthImages == null) exit("Path to ground truth images not set -> stopping.", false);

				System.out.println("coping images..");
				copySmallGroundTruthImages(pathGroundTruthImages, pathResult, thresholdSizeGroundTruth);
				System.out.println("DONE");
			}
			else if (mode == MODE_COPYIMAGES)
			{
				if (pathImagesToCopy == null) exit("Path to images to copy not set -> stopping.", false);
				if (pathDefiningImageNames == null) exit("Path to images defining image names to copy not set -> stopping.", false);

				System.out.println("coping files..");
				FileUtil.copyDefinedFiles(pathImagesToCopy, pathResult, pathDefiningImageNames);
				System.out.println("DONE");
			}
			else if (mode == MODE_EVALUATE)
			{
				if (pathSaliencyMapImages == null) exit("Path to saliency map images not set -> stopping.", false);
				if (pathGroundTruthImages == null && pathGroundTruthFile == null) exit("Path to ground truth images/file not set -> stopping.", false);

				File directorySaliencyMaps = new File(pathSaliencyMapImages);
				if (!directorySaliencyMaps.exists()) exit("Directory '"+directorySaliencyMaps.getAbsolutePath()+"' does not exist -> stopping.", false);
				if (!directorySaliencyMaps.isDirectory()) exit("'"+directorySaliencyMaps.getAbsolutePath()+"' is not a directory -> stopping.", false);
				
				File[] saliencyMapFiles = directorySaliencyMaps.listFiles();
				
				// index 1: number of file
				// index 2: binary threshold between [0; 255]
				// index 3: precision/recall
				double[][][] result = null;

				relativeGtSizes = new HashMap<String, Double>();

				File directoryResults = new File(pathResult);
				if (!directoryResults.exists()) directoryResults.mkdirs();

				if (pathGroundTruthImages != null) result = evaluateWithGroundTruthImages(pathGroundTruthImages, saliencyMapFiles);
				else if (pathGroundTruthFile != null)
				{
					String pathToSaveGroundTruthImages = null;
					if (saveGroundTruthImages) pathToSaveGroundTruthImages = pathResult;
					result = evaluateWithGroundTruthFile(pathGroundTruthFile, saliencyMapFiles, thresholdBinGroundTruth, pathToSaveGroundTruthImages);
				}

				Long t = System.currentTimeMillis();
						
				System.out.print("Saving complete results to '"+pathResult+"/"+"result_all.txt'");
				t = System.currentTimeMillis();
				saveAllResultsToFile(pathResult+"/"+"result_all.txt", result);
				System.out.println(" - finished in "+(System.currentTimeMillis()-t)+"ms");

				System.out.print("Saving ground truth sizes to '"+pathResult+"/"+"ground_truth_sizes.txt'");
				t = System.currentTimeMillis();
				double meanGtSize = saveGtSizesToFile(pathResult+"/"+"ground_truth_sizes.txt");
				System.out.println(" - finished in "+(System.currentTimeMillis()-t)+"ms");
				System.out.println("MEAN GT SIZE: "+meanGtSize);

				System.out.print("Saving mean results to '"+pathResult+"/"+"result_mean.txt'");
				t = System.currentTimeMillis();
				saveMeanResultsToFile(pathResult+"/"+"result_mean.txt", result);
				System.out.println(" - finished in "+(System.currentTimeMillis()-t)+"ms");

				System.out.print("Creating plots");
				t = System.currentTimeMillis();
				plotResult(result, pathResult);
				System.out.println(" - finished in "+(System.currentTimeMillis()-t)+"ms");

				System.out.println("DONE");
			}
		}
		catch (Exception e)
		{
			e.printStackTrace();
		}
		
	}


	/**
	 * Runs evaluation with ground truth descriptions contained in a text file.
	 *
	 * @param pathGroundTruthFile     path to a text file containing ground truth descriptions, for more info see {@link GroundTruthDescription}
	 * @param saliencyMapFiles        array of image files representing saliency maps
	 * @param thresholdBinGroundTruth threshold to determine binary ground truth masks with, in [0.0, 1.0]
	 * @param pathResult              path to save results to
	 *
	 * @return                        result of evaluation:<br>
	 *												index 1: number of file/image<br>
	 *												index 2: binary threshold used in saliency maps, in [0; 255]<br>
	 *												index 3: either {@link GroundTruth#PRECISION} or {@link GroundTruth#RECALL}
	 */
	private static double[][][] evaluateWithGroundTruthFile(String pathGroundTruthFile, File[] saliencyMapFiles, double thresholdBinGroundTruth, String pathResult)
	{
		String groundTruthDescription = FileUtil.readFile(new File(pathGroundTruthFile));
		if (groundTruthDescription == null) exit("Could not read '"+pathGroundTruthFile+"' -> stopping.", false);

		System.out.println("Ground truth descriptions read.");
		String[] descriptions = groundTruthDescription.split("\n\n");

		double[][][] result = new double[descriptions.length][][];
		long t;

		for (int iDescr=0; iDescr<descriptions.length; iDescr++)
		{
			t = System.currentTimeMillis();

			GroundTruthDescription gtd = new GroundTruthDescription(descriptions[iDescr]);

			if (gtd.isValid())
			{
				System.out.print("Evaluating '"+gtd.imageName+"' ("+(iDescr+1)+"/"+descriptions.length+")");

				File sMapFile = FileUtil.findFile(FileUtil.getFileNameWithoutEnding(gtd.imageName), saliencyMapFiles);
				if (sMapFile != null)
				{
					//TEST
					//System.out.println(gtd);

					GroundTruth gt = new GroundTruth(gtd);
					relativeGtSizes.put(gt.imageName, new Double(gt.getRelativeObjectSize()));
					gt.setBinaryThreshold(thresholdBinGroundTruth);

					if (pathResult != null) ImageUtil.saveImageAsPng(gt.getBinaryImage(), pathResult+"/"+gt.imageName);

					SaliencyMap sMap = new SaliencyMap(sMapFile, gt.getWidth(), gt.getHeight());

					result[iDescr] = gt.evaluate(sMap);

					System.out.println(" - finished in "+(System.currentTimeMillis()-t)+"ms");
				}
				else
				{
					System.out.println(" - no saliency map file found.");
				}
			}
			else
			{
				System.out.println("'"+descriptions[iDescr]+"' is not a valid ground truth description -> skipping");
			}
		}

		return result;
	}


	/**
	 * Runs evaluation with ground truths represented by binary images.
	 *
	 * @param pathGroundTruthImages path to a folder containing binary ground truth images
	 * @param saliencyMapFiles      array of image files representing saliency maps
	 *
	 * @return                      result of evaluation:<br>
	 *											  index 1: number of file/image<br>
	 *                                 index 2: binary threshold used in saliency maps, in [0; 255]<br>
	 *                                 index 3: either {@link GroundTruth#PRECISION} or {@link GroundTruth#RECALL}
	 */
	private static double[][][] evaluateWithGroundTruthImages(String pathGroundTruthImages, File[] saliencyMapFiles)
	{
		File fileGroundTruth = new File(pathGroundTruthImages);
		if (!fileGroundTruth.exists()) exit("Directory '"+fileGroundTruth.getAbsolutePath()+"' does not exist.", false);
		if (!fileGroundTruth.isDirectory()) exit("'"+fileGroundTruth.getAbsolutePath()+"' is not a directory.", false);

		long t;

		File[] groundTruthFiles = fileGroundTruth.listFiles();

		double[][][] result = new double[groundTruthFiles.length][][];

		int iFile = 0;

		for (File file: groundTruthFiles)
		{
			System.out.print("Evaluating '"+file.getName()+"' ("+(iFile+1)+"/"+groundTruthFiles.length+")");

			t = System.currentTimeMillis();

			GroundTruth gt = new GroundTruth(file);

			File sMapFile = FileUtil.findFile(FileUtil.getFileNameWithoutEnding(file.getName()), saliencyMapFiles);
			if (sMapFile != null)
			{
				relativeGtSizes.put(gt.imageName, new Double(gt.getRelativeObjectSize()));

				SaliencyMap sMap = new SaliencyMap(sMapFile, gt.getWidth(), gt.getHeight());

				//ImageUtil.saveImageAsPng(sMap.getMap(), "images/test/"+sMap.imageName);

				result[iFile++] = gt.evaluate(sMap);

				System.out.println(" - finished in "+(System.currentTimeMillis()-t)+"ms");
			}
			else
			{
				System.out.println(" - no matching saliency map image found.");
			}
		}

		return result;
					
	}


	/**
	 * Saves plots of an evaluation result to images files.<br>
	 * Two plots are created:<br>
	 * 1. A recall - precision curve (mean values per threshold (saved to file 'plot')<br>
	 * 2. A plot of all recall - precision values (saved to file 'plot_all')
	 * 
	 * @param resultAll        evaluation result<br>
	 *										index 1: number of file/image<br>
	 *										index 2: binary threshold used in saliency maps, in [0; 255]<br>
	 *										index 3: either {@link GroundTruth#PRECISION} or {@link GroundTruth#RECALL}
	 * @param directoryResults folder to save plots to
	 */
	public static void plotResult(double[][][] resultAll, String directoryResults)
	{
		// plot mean values per threshold
		double[][] values = determineMeanPerThreshold(resultAll);

		Plot plot = new Plot();
		plot.setTitle("Evaluation result by threshold");
		plot.setSize(800, 600);
		plot.setXLabel("recall");
		plot.setXRange(0.0, 1.0);
		plot.setYLabel("precision");
		plot.setYRange(0.2, 0.9);
		plot.setMarksStyle("none");

		for (int i=0; i<values.length; i++)	plot.addPoint(0, values[i][GroundTruth.RECALL], values[i][GroundTruth.PRECISION], true);

		BufferedImage plotImage = new BufferedImage(plot.getWidth(), plot.getHeight(), BufferedImage.TYPE_INT_RGB);
		Graphics2D graphics = plotImage.createGraphics();
		plot.paint(graphics);
		graphics.dispose();
		ImageUtil.saveImageAsPng(plotImage, directoryResults+"/plot");


		// plot ALL values
		
		Plot plotAll = new Plot();
		plotAll.setTitle("Evaluation result - ALL values");
		plotAll.setSize(800, 600);
		plotAll.setXLabel("recall");
		plotAll.setXRange(0.0, 1.0);
		plotAll.setYLabel("precision");
		plotAll.setYRange(0, 1);
		plotAll.setXLabel("recall");
		plotAll.setYLabel("precision");
		plotAll.setMarksStyle("pixels");

		for (int iImage=0; iImage<resultAll.length; iImage++)
		{
			if (resultAll[iImage] != null)
			{
				for (int iThreshold=0; iThreshold<SaliencyMap.NUM_GREYSCALES; iThreshold++)
				{
					plotAll.addPoint(0, resultAll[iImage][iThreshold][GroundTruth.RECALL], resultAll[iImage][iThreshold][GroundTruth.PRECISION], false);
				}
			}
		}
				
		BufferedImage plotImageAll = new BufferedImage(plotAll.getWidth(), plotAll.getHeight(), BufferedImage.TYPE_INT_RGB);
		graphics = plotImageAll.createGraphics();
		plotAll.paint(graphics);
		graphics.dispose();
		ImageUtil.saveImageAsPng(plotImageAll, directoryResults+"/plot_all");

		/*
		PlotFrame frame = new PlotFrame("Evaluation result by threshold", plot);
		frame.setSize(800, 600);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.setVisible(true);
		*/
		
	}


	/**
	 * Copies all binary ground truth images whose salient object's relative size to the size of the associated image
	 * is smaller than a threshold.
	 *
	 * @param sourceDir      source folder containing binary ground truth images
	 * @param destinationDir folder to copy images to
	 * @param threshold      maximum relative size of salient object
	 */
	private static void copySmallGroundTruthImages(String sourceDir, String destinationDir, double threshold)
	{
		File directoryGroundTruth = new File(sourceDir);
		if (!directoryGroundTruth.exists()) exit("Directory '"+directoryGroundTruth.getAbsolutePath()+"' does not exist -> stopping.", false);
		if (!directoryGroundTruth.isDirectory()) exit("'"+directoryGroundTruth.getAbsolutePath()+"' is not a directory -> stopping.", false);
		
		File directoryResults = new File(destinationDir);
		if (!directoryResults.exists()) directoryResults.mkdirs();

		File[] groundTruthFiles = directoryGroundTruth.listFiles();
		int nFilesProcessed = 0;
		int nFilesCopied    = 0;

		for (File file: groundTruthFiles)
		{
			GroundTruth gt = new GroundTruth(file);
			if (gt.getRelativeObjectSize() < threshold)
			{
				FileUtil.copyFile(file, new File(directoryResults.getAbsolutePath()+"/"+file.getName()));
				nFilesCopied++;
			}

			nFilesProcessed++;

			if (nFilesProcessed%100 == 0) System.out.println("processed "+nFilesProcessed+" of "+groundTruthFiles.length+" (copied: "+nFilesCopied+")");
		}

		System.out.println("Copied "+nFilesCopied+" files of "+groundTruthFiles.length);
	}


	/**
	 * Writes relative ground truth sizes of all currently loaded ground truths to a file.
	 *
	 * @param fileName name of the file to save to
	 *
	 * @return         average size of relative ground truth size
	 */
	public static double saveGtSizesToFile(String fileName)
	{
		double result = 0.0;

		try
		{
			BufferedWriter writer = new BufferedWriter(new FileWriter(fileName));
			writer.append("# This file shows sizes of ground truths relative to their image's size");
			writer.newLine();
			writer.append("#");
			writer.newLine();
			writer.append("# image_name relative_size_gt");
			writer.newLine();

			double currRelSize;
			Set<String> imageNames = relativeGtSizes.keySet();
			for (String currImageName: imageNames)
			{
				currRelSize = relativeGtSizes.get(currImageName);
				result += currRelSize;
				
				writer.append(currImageName+" "+String.format(Locale.ENGLISH, "%.4g", currRelSize));
				writer.newLine();
			}
			
			writer.close();
		}
		catch (IOException e)
		{
			System.err.println("Could not write to '"+fileName+"': "+e.toString());
			result = -1.0;
		}

		result /= relativeGtSizes.size();

		return result;
	}


	/**
	 * Saves results of an evaluation to a text file.<br>
	 * For each threshold all precision-recall pairs are listed in a row:<br>
	 * <code>threshold precision recall precision recall ...</code>
	 *
	 * @param fileName name of the file to save to
	 * @param result   evaluation result to save<br>
	 *							index 1: number of file/image<br>
	 *							index 2: binary threshold used in saliency maps, in [0; 255]<br>
	 *						   index 3: either {@link GroundTruth#PRECISION} or {@link GroundTruth#RECALL}
	 */
	public static void saveAllResultsToFile(String fileName, double[][][] result)
	{
		try
		{
			BufferedWriter writer = new BufferedWriter(new FileWriter(fileName));
			writer.append("# This file shows results of an evaluation of saliencymaps as described in");
			writer.newLine();
			writer.append("# R. Achanta, S. Hemami, F. Estrada and S. Süsstrunk, Frequency-tuned Salient Region Detection, IEEE International Conference on Computer Vision and Pattern Recognition (CVPR), 2009.");
			writer.newLine();
			writer.append("#");
			writer.newLine();
			writer.append("# threshold precision recall precision recall ...");
			writer.newLine();

			for (int iThreshold=0; iThreshold<SaliencyMap.NUM_GREYSCALES; iThreshold++)
			{
				writer.append(iThreshold+" ");

				for (int iImage=0; iImage<result.length; iImage++)
				{
					if (result[iImage] != null)
					{
						writer.append(String.format(Locale.ENGLISH, "%.4g", result[iImage][iThreshold][GroundTruth.PRECISION])+" "+String.format(Locale.ENGLISH, "%.4g", result[iImage][iThreshold][GroundTruth.RECALL])+" ");
					}
				}

				writer.newLine();
			}

			writer.close();

		}
		catch (IOException e)
		{
			System.err.println("Could not write to '"+fileName+"': "+e.toString());
		}
	}


	/**
	 * Saves results of an evaluation to a text file.<br>
	 *	For each threshold mean precision and mean recall are written to a row:<br>
	 * <code>threshold mean_precision mean_recall</code>
	 *
	 * @param fileName  name of the file to save to
	 * @param resultAll evaluation result<br>
	 *										index 1: number of file/image<br>
	 *										index 2: binary threshold used in saliency maps, in [0; 255]<br>
	 *										index 3: either {@link GroundTruth#PRECISION} or {@link GroundTruth#RECALL}
	 */
	public static void saveMeanResultsToFile(String fileName, double[][][] resultAll)
	{
		try
		{
			double[][] values = determineMeanPerThreshold(resultAll);

			BufferedWriter writer = new BufferedWriter(new FileWriter(fileName));
			writer.append("# This file shows results of an evaluation of saliencymaps as described in");
			writer.newLine();
			writer.append("# R. Achanta, S. Hemami, F. Estrada and S. Süsstrunk, Frequency-tuned Salient Region Detection, IEEE International Conference on Computer Vision and Pattern Recognition (CVPR), 2009.");
			writer.newLine();
			writer.append("#");
			writer.newLine();
			writer.append("# threshold mean_precision mean_recall");
			writer.newLine();

			for (int iThreshold=0; iThreshold<SaliencyMap.NUM_GREYSCALES; iThreshold++)
			{
				writer.append(iThreshold+" ");

				writer.append(String.format(Locale.ENGLISH, "%.4g", values[iThreshold][GroundTruth.PRECISION])+" "+String.format(Locale.ENGLISH, "%.4g", values[iThreshold][GroundTruth.RECALL])+" ");
				
				writer.newLine();
			}

			writer.close();

		}
		catch (IOException e)
		{
			System.err.println("Could not write to '"+fileName+"': "+e.toString());
		}
	}


	/**
	 * Determines mean precision and mean recall for each threshold of an evaluation result.
	 *
	 * @param resultAll evaluation result<br>
	 *						      index 1: number of file/image<br>
	 *								index 2: binary threshold used in saliency maps, in [0; 255]<br>
	 *								index 3: either {@link GroundTruth#PRECISION} or {@link GroundTruth#RECALL}
	 *
	 * @return          means per threshold<br>
	 *                     index 1: threshold, in [0; 255]
	 *                     index 2: either {@link GroundTruth#PRECISION} or {@link GroundTruth#RECALL}
	 */
	private static double[][] determineMeanPerThreshold(double[][][] resultAll)
	{
		double[][] result = new double[SaliencyMap.NUM_GREYSCALES][2];

		// determine actual number of evaluated images
		int actualResults = 0;
		for (int iImage=0; iImage<result.length; iImage++) if (resultAll[iImage] != null) actualResults++;

		for (int iImage=0; iImage<result.length; iImage++)
		{
			for (int iThreshold=0; iThreshold<SaliencyMap.NUM_GREYSCALES; iThreshold++)
			{
				if (resultAll[iImage] != null)
				{
					result[iThreshold][GroundTruth.PRECISION] += resultAll[iImage][iThreshold][GroundTruth.PRECISION] / actualResults;
					result[iThreshold][GroundTruth.RECALL]    += resultAll[iImage][iThreshold][GroundTruth.RECALL]    / actualResults;
				}
			}
		}

		return result;
	}


	/**
	 * Determines applications behaviour by analysing command line parameters.
	 *
	 * @param args command line parameters
	 */
	private static void parseCommandLineParameters(String[] args)
	{
		// set default values
		mode = MODE_EVALUATE;
		pathGroundTruthImages = null;
		pathGroundTruthFile = null;

		if (args.length > 0)
		{
			if (    args[0].startsWith("/?")
			     || args[0].toLowerCase().startsWith("/h")
				  || args[0].toLowerCase().startsWith("-h")
				  || args[0].toLowerCase().startsWith("--h"))
			{
				printUsage();
				System.exit(0);
			}
		}

		for (int iArg=0; iArg<args.length; iArg++)
		{
			try
			{
				int pos = args[iArg].indexOf("=");
				if (pos != -1)
				{
					String var = args[iArg].substring(0, pos);
					String val = args[iArg].substring(pos+1);

					if (var.equalsIgnoreCase("mode"))
					{
							  if (val.equalsIgnoreCase("copySmallGroundTruth")) mode = MODE_COPYSMALLGROUNDTRUTH;
						else if (val.equalsIgnoreCase("copyImages"))           mode = MODE_COPYIMAGES;
						else if (val.equalsIgnoreCase("evaluate"))             mode = MODE_EVALUATE;
					}
					else if (var.equalsIgnoreCase("pathGT"))
					{
						pathGroundTruthImages = val;
					}
					else if (var.equalsIgnoreCase("fileGT"))
					{
						pathGroundTruthFile = val;
					}
					else if (var.equalsIgnoreCase("pathSM"))
					{
						pathSaliencyMapImages = val;
					}
					else if (var.equalsIgnoreCase("pathResult"))
					{
						pathResult = val;
					}
					else if (var.equalsIgnoreCase("thresholdGT"))
					{
						thresholdBinGroundTruth = Double.parseDouble(val);
					}
					else if (var.equalsIgnoreCase("thresholdSize"))
					{
						thresholdSizeGroundTruth = Double.parseDouble(val);
					}
					else if (var.equalsIgnoreCase("pathIm"))
					{
						pathImagesToCopy = val;
					}
					else if (var.equalsIgnoreCase("pathDef"))
					{
						pathDefiningImageNames = val;
					}
					else if (var.equalsIgnoreCase("saveGT"))
					{
						saveGroundTruthImages = Boolean.parseBoolean(val);
					}
					else
					{
						exit("'"+var+"' is not a valid parameter!", true);
					}
				}
				else
				{
					exit("Could not interpret '"+args[iArg]+"'!", true);
				}
			}
			catch (Exception e)
			{
				exit("Could not interpret '"+args[iArg]+"': "+e+".", true);
			}
		}
	}


	/**
	 * Prints instructions on how to use the application to the screen.
	 */
	private static void printUsage()
	{
		String usage =
		"\n" +
		"Saliency Evaluation Tool - USAGE:\n" +
		"\n" +
		"java -jar SaliencyEvaluationTool.jar <parameter1> <parameter2> ...\n" +
		"\n" +
		"1) Evaluation with ground truth images:\n" +
		"      required parameters:\n" +
		"         pathGT=<pathToGroundTruthImages>\n" +
		"         pathSM=<pathToSaliencyMapImages>\n" +
		"      optional parameters:\n" +
		"         mode=Evaluation                  (default: Evaluation)\n" +
		"         pathResult=<pathToSaveResultsTo> (default: current directory)\n" +
		"\n" +
		"2) Evaluation with ground truth descriptions stored in file:\n" +
		"      required parameters:\n" +
		"         fileGT=<pathToGroundTruthFile>\n" +
		"         pathSM=<pathToSaliencyMapImages>\n" +
		"      optional parameters:\n" +
		"         mode=Evaluation                    (default: Evaluation)\n" +
		"         pathResult=<pathToSaveResultsTo>   (default: current directory)\n" +
		"         thresholdGT=<thresholdGroundTruth> (threshold to determine binary ground truth, in [0.0; 1.0], default: 0.5)\n" +
		"         saveGT=true                        (saves binary ground truth images, default: false)\n" +
		"\n" +
		"3) Copy ground truth images with small objects (relative size < thresholdSize)\n" +
		"      required parameters:\n" +
		"         mode=CopySmallGroundTruth\n" +
		"         pathGT=<pathToGroundTruthImages>\n" +
		"      optional parameters:\n" +
		"         pathResult=<pathToCopyTo> (default: current directory)\n" +
		"         thresholdSize=<threshold> (in [0.0; 1.0], default: 0.5)\n" +
		"\n" +
		"4) Copy images defined by image names existing in a directory\n" +
		"      required parameters:\n" +
		"         mode=CopyImages\n" +
		"         pathIm=<pathToCopyFrom>\n" +
		"         pathDef=<pathDefiningNames>\n" +
		"      optional parameters:\n" +
		"         pathResult=<pathToCopyTo> (default: current directory)\n" +
		"";

		System.out.println(usage);
	}


	/**
	 * Prints an error message to the screen and terminates the application.
	 * Optionally instructions on how to use the application can be printed to the screen.
	 *
	 * @param message   error message to print
	 * @param showUsage flag that determines if usage is to print
	 */
	private static void exit(String message, boolean showUsage)
	{
		System.err.println(message);
		if (showUsage) printUsage();
		System.exit(1);
	}
}
