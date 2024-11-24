import cv2
import dlib
import numpy
import time
from imutils import face_utils

import glob
import os

# --------------------------------
# 1.顔ランドマーク検出の前準備
# --------------------------------
# 顔ランドマーク検出ツールの呼び出し
PREDICTOR_PATH = 'shape_predictor_68_face_landmarks.dat'
PREDICTOR = dlib.shape_predictor(PREDICTOR_PATH)
face_detector = dlib.get_frontal_face_detector()

# --------------------------------
# 2.画像から顔のランドマークを抽出し真顔かどうか判定する関数
# --------------------------------
def magao_judgement(img):
    # 顔検出
    img_gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector(img_gry, 1)

    # 検出した全顔に対して処理
    for face in faces:
      # 顔のランドマーク検出
      landmark = PREDICTOR(img_gry, face)
      # 処理高速化のためランドマーク群をNumPy配列に変換(必須)
      landmark = face_utils.shape_to_np(landmark)

      #print(landmark.shape)

      x1 = landmark[67][1]
      y1 = landmark[67][0]

      x2 = landmark[48][1]
      y2 = landmark[48][0]
          
      x3 = landmark[54][1]
      y3 = landmark[54][0]
      u = numpy.array([x2 - x1, y2 - y1])
      v = numpy.array([x3 - x1, y3 - y1])
      L = abs(numpy.cross(u, v) / numpy.linalg.norm(u))
      print (L)

      #真顔判定 
      if L < 8:
        return True
      else:
        return False

# --------------------------------
# 3.真顔以外の顔を真顔に戻すメソッド(関数)を含むクラス
# --------------------------------

class NoFaces(Exception):
    pass

class Face:
  def __init__(self, image, rect):
    self.image = image
    self.landmarks = numpy.matrix(
      [[p.x, p.y] for p in PREDICTOR(image, rect).parts()]
    )

class BeBean: # BeMagao
  SCALE_FACTOR = 1
  FEATHER_AMOUNT = 11

  # 特徴点のうちそれぞれの部位を表している配列のインデックス
  FACE_POINTS = list(range(17, 68))
  MOUTH_POINTS = list(range(48, 61))
  RIGHT_BROW_POINTS = list(range(17, 22))
  LEFT_BROW_POINTS = list(range(22, 27))
  RIGHT_EYE_POINTS = list(range(36, 42))
  LEFT_EYE_POINTS = list(range(42, 48))
  NOSE_POINTS = list(range(27, 35))
  JAW_POINTS = list(range(0, 17))

  ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS + RIGHT_BROW_POINTS +
    NOSE_POINTS + MOUTH_POINTS)

  # オーバーレイする特徴点
  OVERLAY_POINTS = [LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS]

  COLOR_CORRECT_BLUR_FRAC = 0.7

  def __init__(self, before_after = True):
    self.detector = dlib.get_frontal_face_detector()
    #self._load_beans()
    self.before_after = before_after

  def faces_from_image(self, img):
    """
      画像パスから画像オブジェクトとその画像から抽出した特徴点を読み込む。
      ※ 画像内に顔が1つないし複数検出された場合も、返すので正確には「特徴点配列」の配列を返す
    """
    #image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (img.shape[1] * self.SCALE_FACTOR,
                               img.shape[0] * self.SCALE_FACTOR))

    rects = self.detector(img, 1)

    if len(rects) == 0:
      raise NoFaces
    else:
      print("Number of faces detected: {}".format(len(rects)))

    faces = [Face(img, rect) for rect in rects]
    return faces

  def transformation_from_points(self, t_points, o_points):
    """
      特徴点から回転やスケールを調整する。
      t_points: (target points) 対象の特徴点(入力画像)
      o_points: (origin points) 合成元の特徴点(つまりビーン)
    """

    t_points = t_points.astype(numpy.float64)
    o_points = o_points.astype(numpy.float64)

    t_mean = numpy.mean(t_points, axis = 0)
    o_mean = numpy.mean(o_points, axis = 0)

    t_points -= t_mean
    o_points -= o_mean

    t_std = numpy.std(t_points)
    o_std = numpy.std(o_points)

    t_points -= t_std
    o_points -= o_std

    # 行列を特異分解しているらしい
    # https://qiita.com/kyoro1/items/4df11e933e737703d549
    U, S, Vt = numpy.linalg.svd(t_points.T * o_points)
    R = (U * Vt).T

    return numpy.vstack(
      [numpy.hstack((( o_std / t_std ) * R, o_mean.T - ( o_std / t_std ) * R * t_mean.T )),
      numpy.matrix([ 0., 0., 1. ])]
    )

  def get_face_mask(self, face):
    print(face.image.shape[:2])
    image = numpy.zeros(face.image.shape[:2], dtype = numpy.float64)  #三角形のマスク作成
    for group in self.OVERLAY_POINTS:
      self._draw_convex_hull(image, face.landmarks[group], color = 1)

    image = numpy.array([ image, image, image ]).transpose((1, 2, 0))
    image = (cv2.GaussianBlur(image, (self.FEATHER_AMOUNT, self.FEATHER_AMOUNT), 0) > 0) * 1.0
    image = cv2.GaussianBlur(image, (self.FEATHER_AMOUNT, self.FEATHER_AMOUNT), 0)

    return image

  def warp_image(self, image, M, dshape):
    output_image = numpy.zeros(dshape, dtype = image.dtype)
    cv2.warpAffine(
      image,
      M[:2],
      (dshape[1], dshape[0]),
      dst = output_image, borderMode = cv2.BORDER_TRANSPARENT, flags = cv2.WARP_INVERSE_MAP
    )
    return output_image

  def correct_colors(self, t_image, o_image, t_landmarks):
    """
      対象の画像に合わせて、色を補正する
    """
    blur_amount = self.COLOR_CORRECT_BLUR_FRAC * numpy.linalg.norm(
      numpy.mean(t_landmarks[self.LEFT_EYE_POINTS], axis = 0) -
      numpy.mean(t_landmarks[self.RIGHT_EYE_POINTS], axis = 0)
    )
    blur_amount = int(blur_amount)

    if blur_amount % 2 == 0: blur_amount += 1

    t_blur = cv2.GaussianBlur(t_image, (blur_amount, blur_amount), 0)
    o_blur = cv2.GaussianBlur(o_image, (blur_amount, blur_amount), 0)

    # ゼロ除算を避ける　
    o_blur += (128 * (o_blur <= 1.0)).astype(o_blur.dtype)

    return (o_image.astype(numpy.float64) * t_blur.astype(numpy.float64) / o_blur.astype(numpy.float64))

  def to_bean(self, original, target):
    faces = self.faces_from_image(original)
    magao = self.faces_from_image(target)

    # base_imageに合成していく
    base_image = original.copy()

    delta=[0, 30]
    for face in faces:

      magao_mask = self.get_face_mask(magao[0])

      print(face.landmarks[self.MOUTH_POINTS[0]])
      print(face.landmarks[self.MOUTH_POINTS[6]])
      face.landmarks[self.MOUTH_POINTS[0]]+=delta
      face.landmarks[self.MOUTH_POINTS[6]]+=delta
      print(face.landmarks[self.MOUTH_POINTS[0]])
      print(face.landmarks[self.MOUTH_POINTS[6]])

      M = self.transformation_from_points(
        face.landmarks[self.ALIGN_POINTS],
        magao[0].landmarks[self.ALIGN_POINTS]
      )

      warped_magao_mask = self.warp_image(magao_mask, M, base_image.shape)
      combined_mask = numpy.max(
        [self.get_face_mask(face), warped_magao_mask], axis = 0
      )

      warped_image = self.warp_image(magao[0].image, M, base_image.shape)
      warped_corrected_image = self.correct_colors(base_image, warped_image, face.landmarks)
      base_image = base_image * (1.0 - combined_mask) + warped_corrected_image * combined_mask

      #path, ext = os.path.splitext( os.path.basename(image_path) )
      #cv2.imwrite('outputs/output_' + path + ext, base_image)
      cv2.imwrite('output.jpg', base_image)

      if self.before_after is True:
        before_after = numpy.concatenate((original, base_image), axis = 1)
        cv2.imwrite('before_after.jpg', before_after)
    if faces is None:
      return tagert
    else:
      return base_image

  def _draw_convex_hull(self, image, points, color):
    "指定したイメージの領域を塗りつぶす"

    points = cv2.convexHull(points)
    cv2.fillConvexPoly(image, points, color = color)

if __name__ == '__main__':
  be_bean = BeBean()
  img_magao=cv2.imread('akiyama.jpg')
  # カメラの指定(適切な引数を渡す)
  cap = cv2.VideoCapture(0)

  # カメラ画像の表示 ('q'入力で終了)
  while(True):
    ret, img = cap.read()
    #cv2.imwrite('start.jpg', img)

    # 顔のランドマーク検出(2.の関数呼び出し)
    magao = magao_judgement(img)

    if magao is False:  #真顔で無かった場合 = 真顔以外、笑顔になった場合
      img=be_bean.to_bean(img, img_magao)
      img=img.astype(numpy.uint8)
      magao=False
    else:
      img_magao=img

    #time.sleep(1.0)
    # 結果の表示
    cv2.imshow('img', img)

    # 'q'が入力されるまでループ
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


  # 後処理
  cap.release()
  cv2.destroyAllWindows()
