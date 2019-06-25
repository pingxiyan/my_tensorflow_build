#include <iostream>
#include <inference_engine.hpp>
#include <ext_list.hpp>
#include <chrono>
#include <memory>
#include <inference_engine.hpp>
#include <ie_extension.h>

typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;

#include <string>
#include <utility>
#include <algorithm>
#include <iterator>
#include <map>

#include <inference_engine.hpp>


#include <opencv2/opencv.hpp>
using namespace InferenceEngine;

class CCallIR {
public:
	CCallIR(){};
	~CCallIR(){};

	/**
	 * @brief Set device
	 * @param strDev: CPU, GPU
	 */
	bool setDev(std::string strDev, std::string plugin_path = std::string());

	/**
	 * load IR(only xml file)
	 */
	bool loadIR(const std::string& irXml, const std::string& irBin);

	bool infer(const cv::Mat& img);

	bool inferAsync(const cv::Mat& img){return true;};

	cv::Size getInputSize(){
		return cv::Size(inputW, inputH);
	}
private:
	InferencePlugin plugin;
	CNNNetReader networkReader;
	CNNNetwork network;

	InputsDataMap inputInfo;
	std::string firstLayerName;
	OutputsDataMap outputInfo;
	std::string lastLayerName;
	void setInputInfo();
	void setOutputInfo();
	int inputW = 0;
	int inputH = 0;
	int inputC = 0;

	ExecutableNetwork executableNetwork;
	//std::vector<InferRequest::Ptr> vecInferRequest;
	InferRequest::Ptr inferRequest;

	void putImg2InputBlob(const cv::Mat& img, InferRequest::Ptr request);
	void parsingLastLayer(InferRequest::Ptr rqst);
};


/**
 * @brief Set device
 * @param strDev: CPU, GPU
 */
bool CCallIR::setDev(std::string strDev, std::string plugin_path)
{
	std::string pluginRoot = "/opt/intel/computer_vision_sdk/deployment_tools/inference_engine/lib/ubuntu_16.04/intel64";

	try {
		plugin = PluginDispatcher( { pluginRoot, "../3rd/intel64/", "./3rd/intel64/", "", plugin_path}).getPluginByDevice(strDev);
	} catch (const std::exception& error) {
		std::cerr << "Catch exception: " << error.what() << std::endl;
		return false;
	} catch (...) {
		std::cerr << "Unknown/internal exception happened." << std::endl;
		return false;
	}
	return true;
}

bool CCallIR::loadIR(const std::string& irXml, const std::string& irBin) {
	try {
		/** Read network model **/
		networkReader.ReadNetwork(irXml);

		/** Extract model name and load weigts **/
		networkReader.ReadWeights(irBin);
		network = networkReader.getNetwork();

		Precision p = network.getPrecision();
		std::cout << "network.getPrecision() = " << p.name() << std::endl;

		firstLayerName = network.getInputsInfo().begin()->first;
		std::cout << "firstLayerName = " << firstLayerName << std::endl;

		inputW = network.getInputsInfo().begin()->second->getTensorDesc().getDims()[3];
		inputH = network.getInputsInfo().begin()->second->getTensorDesc().getDims()[2];
		inputC = network.getInputsInfo().begin()->second->getTensorDesc().getDims()[1];

		std::cout << "inputWidth = " << inputW << std::endl;
		std::cout << "inputHeight = " << inputH << std::endl;
		std::cout << "inputC = " << inputC << std::endl;

		network.setBatchSize(1);
		std::cout << "setBatchSize = " << 1 << std::endl;

		// 3. Configure input & output
		setInputInfo();
		setOutputInfo();

		// 4. Loading model to the plugin
		executableNetwork = plugin.LoadNetwork(network, { });

//#define REQUST_NUM 4
//	for (int i = 0; i < REQUST_NUM; i++) {
//		vecInferRequest.push_back(executableNetwork.CreateInferRequestPtr());
//	}
		inferRequest = executableNetwork.CreateInferRequestPtr();
	} catch (const std::exception& error) {
		std::cerr << "Catch exception: " << error.what() << std::endl;
		return 1;
	} catch (...) {
		std::cerr << "Unknown/internal exception happened." << std::endl;
		return 1;
	}
	return true;
}

void CCallIR::setInputInfo() {
	inputInfo = InputsDataMap(network.getInputsInfo());
	if (inputInfo.size() != 1) {
		throw std::logic_error("This sample accepts networks having only one input");
	}

	InputInfo::Ptr& input = inputInfo.begin()->second;
	input->setPrecision(Precision::U8);
	input->getInputData()->setLayout(Layout::NCHW);
}

void CCallIR::setOutputInfo() {
	outputInfo = OutputsDataMap(network.getOutputsInfo());
//    if (outputInfo.size() != 1) {
//        throw std::logic_error("This sample accepts networks having only one output");
//    }
	DataPtr& output = outputInfo.begin()->second;
	lastLayerName = outputInfo.begin()->first;

	std::cout << "last layer name = " << lastLayerName << std::endl;

	output->setPrecision(Precision::FP32);
	output->setLayout(Layout::NCHW);
}

void CCallIR::putImg2InputBlob(const cv::Mat& img, InferRequest::Ptr request) {
	Blob::Ptr inputBlob = request->GetBlob(firstLayerName);
	uint8_t * blob_data = inputBlob->buffer().as<uint8_t*>();

	cv::Mat resized_image(img);
	if (inputW != img.size().width || inputH != img.size().height) {
		cv::resize(img, resized_image, cv::Size(inputW, inputH));
//        std::cout << "resize image to [" << inputW << " ," << inputH << "]" << std::endl;
//        cv::imshow("res", resized_image);
//        cv::waitKey(0);
	}

	for (int c = 0; c < inputC; c++) {
		for (int h = 0; h < inputH; h++) {
			for (int w = 0; w < inputW; w++) {
				blob_data[c * inputW * inputH + h * inputW + w] = resized_image.at<cv::Vec3b>(h, w)[c];
			}
		}
	}
}

bool CCallIR::infer(const cv::Mat& img) {
	auto t0 = std::chrono::high_resolution_clock::now();

	putImg2InputBlob(img, inferRequest);

	inferRequest->StartAsync();

	if (OK != inferRequest->Wait(IInferRequest::WaitMode::RESULT_READY)) {
		std::cout << "inferRequest->Wait FAIL";
	}

	auto t1 = std::chrono::high_resolution_clock::now();
	auto ocv_infer_time = std::chrono::duration_cast<ms>(t1 - t0).count();
	std::cout << "infer time = " << ocv_infer_time << " ms" << std::endl;

	parsingLastLayer(inferRequest);

	return true;
}

void CCallIR::parsingLastLayer(InferRequest::Ptr rqst) {
	Blob::Ptr feaLayer = rqst->GetBlob(lastLayerName);
	const float *fea = feaLayer->buffer().as<float *>();

	std::cout << "==================================" << std::endl;
	for (int i = 0; i < 10; i++) {
		std::cout << i << ", " << fea[i] << std::endl;
	}
}

void test_by_ie(cv::Mat src){
	std::string FLAGS_pp="";
	std::string FLAGS_d="CPU";
	std::string FLAGS_m = "../lenet.xml";
	std::string binFileName = "../lenet.xml";

	CCallIR* callir = new CCallIR();
	callir->setDev("CPU", std::string());
	callir->loadIR("../lenet.xml", "../lenet.bin");
	callir->infer(src);
}

int main(int argc, char** argv) {
	cv::Mat src = cv::imread("../test1.png", 0);
	
	test_by_ie(src);

	cv::imshow("src", src);
	cv::waitKey(0);
	return 1;
}
