import { Component, AfterViewInit, OnInit, ElementRef } from '@angular/core';
import { ViewChild } from '@angular/core';
import { FilterService } from '../../services/filter/filter.service';
import { FrameService } from '../../services/frame/frame.service';
import { Filter } from '../../models/filter';

@Component({
  selector: 'app-display',
  templateUrl: './display.component.html',
  styleUrls: ['./display.component.css']
})
export class DisplayComponent implements AfterViewInit, OnInit {
  @ViewChild('videoElement') videoElement: ElementRef;
  unprocessedCanvas = document.createElement('canvas');
  @ViewChild('processedCanvas') processedCanvas: ElementRef;

  filters: Filter[];
  processed_context: any;
  unprocessed_context: any;
  video_native_element: any;
  videoPlaying = false;
  quality = 0.5;
  sock: any;

  constructor(
    public frameService: FrameService,
    private filterService: FilterService) {
  }

  capture() {
    this.setPlaying();
    if (!this.videoPlaying) {
      return;
    }

    this.unprocessed_context.drawImage(this.video_native_element, 0, 0, this.video_native_element.clientWidth, this.video_native_element.clientHeight, 0, 0, this.unprocessed_context.canvas.width, this.unprocessed_context.canvas.height);
    if (this.shouldSendReques()){
      let post_data = {'frame': this.unprocessedCanvas.toDataURL("image/jpeg", 0.95), 'config': this.filters}
      this.sock.send(JSON.stringify(post_data))
    }
    else{
      this.processed_context.drawImage(this.video_native_element, 0, 0);
      requestAnimationFrame(this.capture.bind(this));
    }
  }

  ngOnInit(): void {
    this.getFilters();
    this.setupConnection();
    this.initCamera({ video: true, audio: true });
  }

  private shouldSendReques (): boolean {
    if (this.quality==0){
      return false;
    }
    for (let i = 0; i < this.filters.length; i++) {
      if (this.filters[i].selectedModel) {
        return true;
      }
    }
    return false;
  }

  private setupConnection() {
    this.sock = this.frameService.openImageConnection();
    this.sock.onmessage = (v) => {
      this.processed_context.drawImage(this.video_native_element, 0, 0);
      let data = JSON.parse(v['data'])
      this.processDetection(data);
      requestAnimationFrame(this.capture.bind(this));
    };
  }

  private processDetection(data: any) {
    if (data['bboxes']) {
      for (let box of data['bboxes']) {
        this.processed_context.beginPath();
        this.processed_context.lineWidth = 3;
        let color = this.toColor(16777216 / (box['category_id'] + 1));
        this.processed_context.strokeStyle = color;
        let bbox = box['bbox'];
        this.processed_context.rect(
          bbox[0] * this.processed_context.canvas.width,
          bbox[1] * this.processed_context.canvas.height,
          bbox[2] * this.processed_context.canvas.width,
          bbox[3] * this.processed_context.canvas.height
        );
        this.processed_context.font = "bold 1.25em Arial";
        this.processed_context.fillStyle = color;
        this.processed_context.fillText(
          box['class'],
          bbox[0] * this.processed_context.canvas.width - 2,
          bbox[1] * this.processed_context.canvas.height - 4);
        this.processed_context.stroke();
      }
    }
  }

  private toColor(num: number): string {
    num >>>= 0;
    var b = num & 0xFF,
        g = (num & 0xFF00) >>> 8,
        r = (num & 0xFF0000) >>> 16;
    return "rgba(" + [r, g, b].join(",") + ")";
}

  getFilters(): void {
    this.filterService.getFilters()
      .subscribe(filters => this.filters = filters);
  }

  ngAfterViewInit(): void {
    this.processed_context = this.processedCanvas.nativeElement.getContext("2d");
    this.unprocessed_context = this.unprocessedCanvas.getContext("2d");
    this.video_native_element = this.videoElement.nativeElement;
    this.setupCamera();
  }

  private setupCamera() {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
        this.video_native_element.srcObject = stream;
      });
    }
  }

  pause() {
    this.video_native_element.pause();
    this.setPlaying();
  }

  resume() {
    if (!this.video_native_element.srcObject?.active){
      this.setupCamera();
    }
    if (this.sock.readyState!=WebSocket.OPEN) {
      this.setupConnection();
    }
    this.video_native_element.play();
    this.capture();
  }

  setPlaying() {
    this.videoPlaying = this.video_native_element.paused==false
      && this.sock.readyState == WebSocket.OPEN
      && this.video_native_element.srcObject.active;
  }

  initCamera(config:any) {
    var browser = <any>navigator;

    browser.getUserMedia = (browser.getUserMedia ||
      browser.webkitGetUserMedia ||
      browser.mozGetUserMedia ||
      browser.msGetUserMedia);

    browser.mediaDevices.getUserMedia(config).then(stream => {
      this.video_native_element.srcObject = stream;
    });
  }

  updateCanvas(){
    this.processedCanvas.nativeElement.width = this.video_native_element.clientWidth;
    this.processedCanvas.nativeElement.height = this.video_native_element.clientHeight;
    this.updateUnprocessedCanvas();
  }

  updateUnprocessedCanvas(){
    this.unprocessedCanvas.height = this.video_native_element.clientHeight * this.quality;
    this.unprocessedCanvas.width = this.video_native_element.clientWidth * this.quality;
  }
}
