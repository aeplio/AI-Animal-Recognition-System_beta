from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import paddlehub as hub
import cv2
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit

# 确保上传文件夹存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 加载PaddleHub模型
try:
    # 加载ResNet50_VD模型
    resnet_model = hub.Module(name='resnet50_vd_animals')
    
    # 加载MobileNetV2模型
    mobilenet_model = hub.Module(name='mobilenet_v2_animals')
    
    print('Models loaded successfully')
except Exception as e:
    print(f'Error loading models: {str(e)}')
    raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        print('Error: No file part')
        return jsonify({'error': '没有文件上传'})
    
    file = request.files['file']
    model_type = request.form.get('model', 'resnet')  # 默认使用ResNet模型
    
    if file.filename == '':
        print('Error: No selected file')
        return jsonify({'error': '没有选择文件'})
    
    if file:
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            print(f'File saved to {filepath}')
            
            # 处理图像
            try:
                # 读取图像
                image = cv2.imread(filepath)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # 选择模型
                model = resnet_model if model_type == 'resnet' else mobilenet_model
                
                # 进行预测
                results = model.classification(images=[image])
                print(f'Raw prediction results: {results}')
                
                # 处理预测结果
                predictions = []
                
                try:
                    # 确保结果是有效的
                    if results and isinstance(results, list) and len(results) > 0:
                        # 处理字典格式的结果
                        if isinstance(results[0], dict):
                            for category, probability in results[0].items():
                                try:
                                    probability = float(probability)
                                    predictions.append({
                                        'category': str(category),
                                        'probability': round(probability * 100, 2)
                                    })
                                except (ValueError, TypeError):
                                    continue
                        # 处理列表格式的结果
                        elif isinstance(results[0], list):
                            for result in results[0]:
                                try:
                                    if isinstance(result, (list, tuple)) and len(result) == 2:
                                        category, probability = result
                                        probability = float(probability)
                                        predictions.append({
                                            'category': str(category),
                                            'probability': round(probability * 100, 2)
                                        })
                                except (ValueError, TypeError):
                                    continue
                                except Exception as e:
                                    print(f'Error processing individual result: {str(e)}')
                                    continue
                    
                    if not predictions:
                        return jsonify({'error': '无法识别图片内容，请尝试使用其他图片'})
                
                    # 按概率排序并获取前5个结果
                    predictions = sorted(predictions, key=lambda x: x['probability'], reverse=True)
                    predictions = predictions[:5]
                    print(f'Final predictions: {predictions}')
                    
                    return jsonify({
                        'success': True,
                        'predictions': predictions
                    })
                except Exception as e:
                    print(f'Error during image processing: {str(e)}')
                    return jsonify({'error': f'图像处理错误: {str(e)}'})
            except Exception as e:
                print(f'Error during file handling: {str(e)}')
                return jsonify({'error': str(e)})

        except Exception as e:
            print(f'Error during file handling: {str(e)}')
            return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)