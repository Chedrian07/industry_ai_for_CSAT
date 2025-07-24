import os
import glob
from pathlib import Path
from pdf2image import convert_from_path
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_pdf_to_png(pdf_path, output_dir):
    """
    PDF 파일을 PNG 이미지로 변환
    
    Args:
        pdf_path (str): PDF 파일 경로
        output_dir (str): PNG 이미지 저장할 디렉토리 경로
    
    Returns:
        bool: 성공 여부
    """
    try:
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"🔄 변환 시작: {Path(pdf_path).name}")
        logger.info(f"📁 저장 위치: {output_dir}")
        
        # PDF를 이미지로 변환 (300 DPI로 고품질 변환)
        pages = convert_from_path(pdf_path, dpi=300)
        
        logger.info(f"📄 총 페이지 수: {len(pages)}")
        
        # 각 페이지를 PNG로 저장
        for i, page in enumerate(pages, 1):
            png_filename = f"page_{i:03d}.png"
            png_path = os.path.join(output_dir, png_filename)
            page.save(png_path, 'PNG')
            
            if i % 10 == 0 or i == len(pages):
                logger.info(f"  ✅ 진행률: {i}/{len(pages)} 페이지 완료")
        
        logger.info(f"🎉 변환 완료: {Path(pdf_path).name} → {len(pages)}개 페이지")
        return True
        
    except Exception as e:
        logger.error(f"❌ 변환 실패 {Path(pdf_path).name}: {str(e)}")
        return False

def find_all_pdfs(base_dir):
    """
    기본 디렉토리에서 모든 PDF 파일 찾기
    
    Args:
        base_dir (Path): 검색할 기본 디렉토리
    
    Returns:
        list: PDF 파일 경로 리스트
    """
    pdf_files = []
    
    # 모든 하위 디렉토리에서 PDF 파일 찾기
    for pdf_path in base_dir.rglob("*.pdf"):
        pdf_files.append(pdf_path)
    
    # 파일명 순으로 정렬
    pdf_files.sort()
    
    logger.info(f"📚 발견된 PDF 파일: {len(pdf_files)}개")
    for pdf in pdf_files:
        logger.info(f"  📄 {pdf.relative_to(base_dir)}")
    
    return pdf_files

def create_output_directory_name(pdf_path, base_dir):
    """
    PDF 파일 경로로부터 출력 디렉토리 이름 생성
    
    Args:
        pdf_path (Path): PDF 파일 경로
        base_dir (Path): 기본 디렉토리
    
    Returns:
        str: 출력 디렉토리 이름
    """
    # PDF 파일명에서 확장자 제거
    pdf_name = pdf_path.stem
    
    # 출력 디렉토리 이름 생성
    return pdf_name

def process_all_pdfs():
    """
    모든 PDF 파일을 처리하여 PNG로 변환
    """
    # 현재 스크립트가 있는 디렉토리 (dataset 디렉토리)
    base_dir = Path(__file__).parent
    
    # 이미지 저장할 기본 디렉토리
    images_dir = base_dir / "images"
    
    logger.info("🚀 PDF → PNG 변환 작업 시작")
    logger.info(f"📂 기본 디렉토리: {base_dir}")
    logger.info(f"🖼️  이미지 저장 디렉토리: {images_dir}")
    
    # 모든 PDF 파일 찾기
    pdf_files = find_all_pdfs(base_dir)
    
    if not pdf_files:
        logger.warning("⚠️  PDF 파일을 찾을 수 없습니다.")
        return
    
    logger.info(f"\n{'='*60}")
    logger.info(f"📋 처리 계획:")
    logger.info(f"{'='*60}")
    
    # 처리 계획 출력
    for pdf_path in pdf_files:
        output_dir_name = create_output_directory_name(pdf_path, base_dir)
        output_path = images_dir / output_dir_name
        logger.info(f"📄 {pdf_path.name}")
        logger.info(f"   → {output_path.relative_to(base_dir)}/")
        logger.info("")
    
    # 실제 변환 작업 수행
    total_files = len(pdf_files)
    success_count = 0
    
    logger.info(f"{'='*60}")
    logger.info(f"🔄 변환 작업 시작")
    logger.info(f"{'='*60}")
    
    for i, pdf_path in enumerate(pdf_files, 1):
        logger.info(f"\n📊 진행률: {i}/{total_files}")
        logger.info(f"{'─'*40}")
        
        # 출력 디렉토리 경로 생성
        output_dir_name = create_output_directory_name(pdf_path, base_dir)
        output_dir = images_dir / output_dir_name
        
        # PDF 변환 실행
        if convert_pdf_to_png(str(pdf_path), str(output_dir)):
            success_count += 1
        
        logger.info(f"{'─'*40}")
    
    # 최종 결과 요약
    logger.info(f"\n{'='*60}")
    logger.info(f"🎊 작업 완료 요약")
    logger.info(f"{'='*60}")
    logger.info(f"📊 총 처리 파일: {total_files}개")
    logger.info(f"✅ 성공: {success_count}개")
    logger.info(f"❌ 실패: {total_files - success_count}개")
    logger.info(f"📈 성공률: {(success_count/total_files)*100:.1f}%")
    
    if success_count == total_files:
        logger.info(f"🎉 모든 PDF 변환이 성공적으로 완료되었습니다!")
    else:
        logger.warning(f"⚠️  일부 파일 변환에 실패했습니다.")
    
    # 생성된 디렉토리 구조 출력
    if images_dir.exists():
        logger.info(f"\n📁 생성된 이미지 디렉토리 구조:")
        for item in sorted(images_dir.iterdir()):
            if item.is_dir():
                png_files = list(item.glob("*.png"))
                logger.info(f"   📂 {item.name}/ ({len(png_files)}개 이미지)")

def main():
    """메인 함수"""
    try:
        process_all_pdfs()
    except Exception as e:
        logger.error(f"❌ 전체 작업 실패: {str(e)}")
        raise

if __name__ == "__main__":
    main()
