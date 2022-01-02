import { Component, OnInit, HostListener, ElementRef, Output, EventEmitter } from '@angular/core';
import { StateService } from './services/state/state.service';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {

  constructor(
    private stateService: StateService
  ) {}
  ngOnInit() {
    let vh = window.innerHeight * 0.01;
    document.documentElement.style.setProperty('--vh', `${vh}px`);
  }

  @HostListener('window:resize', [])
  onResize() {
    let vh = window.innerHeight * 0.01;
    document.documentElement.style.setProperty('--vh', `${vh}px`);
  }

  contentResize(opened: boolean): void {
    this.stateService.setMenuOpened(opened)
  }
}
