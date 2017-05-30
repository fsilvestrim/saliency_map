
package de.unibonn.informatik.ivs.set;


import java.io.*;
import java.awt.*;
import java.awt.image.*;
import javax.imageio.*;


/**
 * Represents the ground truth for salient objects in an image.<br>
 * The ground truth always refers to a specified image.<br>
 * <br>
 * An instance of GroundTruth can either be constructed by specifying an binary image
 * ({@link #GroundTruth(java.io.File)}, {@link #GroundTruth(java.lang.String)})<br>
 * or by specifing a ground truth description
 * ({@link #GroundTruth(de.unibonn.informatik.ivs.set.GroundTruthDescription)}, see {@link GroundTruthDescription} for details).<br>
 * <br>
 * This class provides methods to evaluate saliency maps (see {@link SaliencyMap}) against it's instances
 * ({@link #evaluate(de.unibonn.informatik.ivs.set.SaliencyMap)}, {@link #evaluate(de.unibonn.informatik.ivs.set.SaliencyMap, int)}).<br>
 * The evaluation method used is defined in<br>
 * R. Achanta, S. Hemami, F. Estrada and S. Süsstrunk, Frequency-tuned Salient Region Detection, IEEE International Conference on Computer Vision and Pattern Recognition (CVPR), 2009.
 *
 * @author		Bernd Wendt
 * @version		2011.0312
 *
 */
public class GroundTruth
{

	/**
	 * Constant for precision value
	 */
	public static final int PRECISION = 0;

	/**
	 * Constant for recall value
	 */
	public static final int RECALL    = 1;


	/**
	 * Name of the image ground truth refers to
	 */
	String imageName;

	/**
	 * Ground truth represented by an binary image.
	 */
	BufferedImage binaryGroundTruth;

	/**
	 * Array representing ground truth as probability values in [0.0; 1.0]
	 */
	float[][] greyGroundTruth;


	/**
	 * Constructs ground truth from an binary image.
	 *
	 * @param file File containing the image
	 */
	public GroundTruth(File file)
	{
		loadGroundTruthImage(file);
	}


	/**
	 * Constructs ground truth from an binary image
	 *
	 * @param fileName File name of the image
	 */
	public GroundTruth(String fileName)
	{
		loadGroundTruthImage(fileName);
	}


	/**
	 * Constructs ground truth from a description
	 *
	 * @param description Description of ground truth
	 */
	public GroundTruth(GroundTruthDescription description)
	{
		this.imageName = description.imageName;
		greyGroundTruth = new float[description.imageSize.width][description.imageSize.height];

		float greyFactor = 1f / description.rectangles.size();

		for (Rectangle currentRect: description.rectangles)
		{
			// rectangles might exceed image bounds
			if (currentRect.x+currentRect.width  > greyGroundTruth.length)    currentRect.width  = greyGroundTruth.length-currentRect.x;
			if (currentRect.y+currentRect.height > greyGroundTruth[0].length) currentRect.height = greyGroundTruth[0].length-currentRect.y;

			for (int x=currentRect.x; x<currentRect.x+currentRect.width; x++)
			{
				for (int y=currentRect.y; y<currentRect.y+currentRect.height; y++)
				{
					greyGroundTruth[x][y] += greyFactor;
				}
			}
		}
	}


	/**
	 * Calculates binary image of ground truth from probability values and a threshold.
	 * Only probability values above the threshold are set in binary image.
	 *
	 * This method does nothing if ground truth was not constructed from a ground truth description.
	 *
	 * @param threshold The threshold
	 */
	public void setBinaryThreshold(double threshold)
	{
		if (greyGroundTruth == null) return;

		if (threshold < 0f) threshold = 0f;
		else if (threshold > 1f) threshold = 1f;

		binaryGroundTruth = new BufferedImage(greyGroundTruth.length, greyGroundTruth[0].length, BufferedImage.TYPE_BYTE_BINARY);

		for (int x=0; x<greyGroundTruth.length; x++)
		{
			for (int y=0; y<greyGroundTruth[x].length; y++)
			{
				if (greyGroundTruth[x][y] > threshold) binaryGroundTruth.setRGB(x, y, 0xFFFFFFFF);
			}
		}
	}


	/**
	 * Loads ground truth from an image file
	 *
	 * @param fileName File name (path) of image to load
	 */
	private void loadGroundTruthImage(String fileName)
	{
		File file = new File(fileName);
		loadGroundTruthImage(file);
	}


	/**
	 * Loads ground truth from an image file
	 *
	 * @param file Image file to load
	 */
	private void loadGroundTruthImage(File file)
	{
		if (!file.exists())
		{
			System.err.println("File '"+file.getAbsolutePath()+"' does not exist.");
			return;
		}
		else if (file.isDirectory())
		{
			System.err.println("File '"+file.getAbsolutePath()+"' is a directory.");
			return;
		}

		try
		{
			binaryGroundTruth = ImageIO.read(file);

			// image name is file name without ending
			imageName = file.getName();
			int pos = imageName.lastIndexOf('.');
			if (pos != -1) imageName = imageName.substring(0, pos);

		}
		catch (IOException e)
		{
			System.err.println("Could not load image '"+file.getAbsolutePath()+"': "+e.toString());
		}
	}


	/**
	 * Calculates relative size of ground truth object to image size
	 *
	 * @return relative ground truth size in [0.0; 1.0]
	 */
	public double getRelativeObjectSize()
	{
		if (binaryGroundTruth == null) return -1.0;

		int i = 0;

		for (int x=0; x<binaryGroundTruth.getWidth(); x++)
		{
			for (int y=0; y<binaryGroundTruth.getHeight(); y++)
			{
				if (binaryGroundTruth.getRGB(x, y) == 0xFFFFFFFF) i++;
			}
		}

		return (double)i / (binaryGroundTruth.getWidth()*binaryGroundTruth.getHeight());
	}


	/**
	 * Calculates precision/recall values for a saliency map depending on this ground truth.
	 * A binary threshold must be given that determines witch values in the (greyscaled) saliency map
	 * are to be taken into consideration.
	 *
	 * @param saliencyMap     Saliency map to evaluate
	 * @param binaryThreshold threshold for saliency values to consider
	 *
	 * @return                Array containing precision/recall (allowed indices: {@link #PRECISION} and {@link #RECALL})
	 */
	public double[] evaluate(SaliencyMap saliencyMap, int binaryThreshold)
	{
		if (binaryGroundTruth == null) return null;

		if (    saliencyMap.getWidth()  != binaryGroundTruth.getWidth()
		     || saliencyMap.getHeight() != binaryGroundTruth.getHeight()) return null;

		int[] saliencyValues = saliencyMap.getMapValues();

		int truePositive  = 0;
		int falsePositive = 0;
		int falseNegative = 0;

		int i = 0;

		for (int x=0; x<binaryGroundTruth.getWidth(); x++)
		{
			for (int y=0; y<binaryGroundTruth.getHeight(); y++)
			{
				if (binaryGroundTruth.getRGB(x, y) == 0xFFFFFFFF)
				{
					if (saliencyValues[i] >= binaryThreshold) truePositive++;
					else falseNegative++;
				}
				else if (saliencyValues[i] >= binaryThreshold) falsePositive++;

				i++;
			}
		}

		double[] result = new double[2];
		result[PRECISION] = (double)truePositive / (truePositive+falsePositive);
		result[RECALL   ] = (double)truePositive / (truePositive+falseNegative);

		return result;
	}


	/**
	 * Calculates precision/recall values for a saliency map and each binary threshold depending on this ground truth.
	 * Each possible grey value of the saliency map is considered as threshold to determine the salient object, then
	 * precision/recall are calculated.
	 *
	 * @param saliencyMap Saliency map to evaluate
	 *
	 * @return            Array containing precision/recall for each greyvalue threshold<br>
	 *                    index1: theshold in [0; 255]<br>
	 *                    index2: {@link #PRECISION} and {@link #RECALL}
	 */
	public double[][] evaluate(SaliencyMap saliencyMap)
	{
		if (    saliencyMap.getWidth()  != binaryGroundTruth.getWidth()
		     || saliencyMap.getHeight() != binaryGroundTruth.getHeight())
		{
			System.err.println("Cannot evaluate '"+imageName+"': size of saliency map does not match");
			return null;
		}

		int[] saliencyValues = saliencyMap.getMapValues();
		if (saliencyValues == null)
		{
			System.err.println("Cannot evaluate '"+imageName+"': saliency map values not available");
			return null;
		}

		boolean[] groundTruthValues = getValues();
		if (groundTruthValues == null)
		{
			System.err.println("Cannot evaluate '"+imageName+"': ground truth values not available");
			return null;
		}

		double[][] result = new double[SaliencyMap.NUM_GREYSCALES][2];

		int nGroundTruth, nSaliency, nMatch;
		
		for (int binaryThreshold=0; binaryThreshold<SaliencyMap.NUM_GREYSCALES; binaryThreshold++)
		{
			nGroundTruth = nSaliency = nMatch = 0;

			// TEST
			//ImageUtil.saveImageAsPng(saliencyMap.getBinaryMap(binaryThreshold), "images/binary_saliency_maps/"+saliencyMap.imageName+"_"+binaryThreshold);
			
			for (int i=0; i<groundTruthValues.length; i++)
			{
				if (groundTruthValues[i])
				{
					nGroundTruth++;

					if (saliencyValues[i] >= binaryThreshold)
					{
						nSaliency++;
						nMatch++;
					}
				}
				else if (saliencyValues[i] >= binaryThreshold) nSaliency++;
			}

			if (nSaliency    > 0) result[binaryThreshold][PRECISION] = (double)nMatch / nSaliency;
			if (nGroundTruth > 0) result[binaryThreshold][RECALL   ] = (double)nMatch / nGroundTruth;

			//System.out.println("binaryThreshold = "+binaryThreshold+": nMatch = "+nMatch+", nSaliency = "+nSaliency+", nGroundTruth = "+nGroundTruth);
		}

		return result;
	}


	/**
	 * Returns binary ground truth values in an one-dimensional array.
	 *
	 * @return binary ground truth values
	 */
	public boolean[] getValues()
	{
		if (binaryGroundTruth == null) return null;

		boolean[] result = new boolean[binaryGroundTruth.getWidth()*binaryGroundTruth.getHeight()];
		int i=0;

		for (int x=0; x<binaryGroundTruth.getWidth(); x++)
		{
			for (int y=0; y<binaryGroundTruth.getHeight(); y++)
			{
				if (binaryGroundTruth.getRGB(x, y) == 0xFFFFFFFF) result[i] = true;
				i++;
			}
		}

		return result;
	}


	/**
	 * Returns binary ground truth as an image
	 *
	 * @return binary ground truth image
	 */
	public BufferedImage getBinaryImage()
	{
		return binaryGroundTruth;
	}
	

	/**
	 * Calculates F1-measure from precision and recall.
	 * The F1-measure is the harmonic mean of precision and recall.
	 *
	 * @param precisionRecall Array containing precision/recall (indices: {@link #PRECISION} and {@link #RECALL})
	 *
	 * @return                F1-measure
	 */
	public static double getF1Measure(double[] precisionRecall)
	{
		return getFMeasure(precisionRecall, 1.0);
	}


	/**
	 * Calculates F-measure from precision and recall with given beta value as defined in<br>
	 * Nancy Chinchor, MUC-4 Evaluation Metrics, in Proc. of the Fourth Message Understanding Conference, pp. 22–29, 1992.<br>
	 * <a href="http://www.aclweb.org/anthology-new/M/M92/M92-1002.pdf">http://www.aclweb.org/anthology-new/M/M92/M92-1002.pdf</a>
	 *
	 * @param precisionRecall Array containing precision/recall (indices: {@link #PRECISION} and {@link #RECALL})
	 * @param beta            Beta value
	 *
	 * @return                F-measure
	 */
	public static double getFMeasure(double[] precisionRecall, double beta)
	{
		double result = (1.0 + Math.pow(beta, 2.0)) * precisionRecall[PRECISION] * precisionRecall[RECALL];
		result /= Math.pow(beta, 2.0) * precisionRecall[PRECISION] + precisionRecall[RECALL];

		return result;
	}


	/**
	 * Returns the width of the image ground truth refers to.
	 *
	 * @return image width
	 */
	public int getWidth()
	{
		if (binaryGroundTruth != null) return binaryGroundTruth.getWidth();
		return -1;
	}


	/**
	 * Returns the height of the image ground truth refers to.
	 *
	 * @return image height
	 */
	public int getHeight()
	{
		if (binaryGroundTruth != null) return binaryGroundTruth.getHeight();
		return -1;
	}

	
}
