package data

import (
	"fmt"
	"github.com/pygo/go/tensor"
	"os"
	"path/filepath"
)

type ImageNetDataset struct {
	Root, Split string
	Transform   Transform
	Classes     []string
	Samples     []struct {
		path     string
		classIdx int
	}
}

func NewImageNet(root, split string, transform Transform) (*ImageNetDataset, error) {
	ds := &ImageNetDataset{Root: root, Split: split, Transform: transform}
	entries, err := os.ReadDir(filepath.Join(root, split))
	if err != nil {
		return nil, err
	}
	cm := map[string]int{}
	for i, e := range entries {
		if e.IsDir() {
			ds.Classes = append(ds.Classes, e.Name())
			cm[e.Name()] = i
		}
	}
	for cls, idx := range cm {
		imgs, _ := filepath.Glob(filepath.Join(root, split, cls, "*.JPEG"))
		for _, img := range imgs {
			ds.Samples = append(ds.Samples, struct {
				path     string
				classIdx int
			}{img, idx})
		}
	}
	fmt.Printf("ImageNet %s: %d classes %d images\n", split, len(ds.Classes), len(ds.Samples))
	return ds, nil
}
func (d *ImageNetDataset) Len() int { return len(d.Samples) }
func (d *ImageNetDataset) Get(idx int) *Batch {
	s := d.Samples[idx]
	img := tensor.Zeros(tensor.Shape{3, 224, 224})
	if d.Transform != nil {
		img = d.Transform(img)
	}
	return &Batch{Inputs: []*tensor.Tensor{img}, Targets: []*tensor.Tensor{tensor.FromSlice([]int64{int64(s.classIdx)}, tensor.Shape{1})}}
}
