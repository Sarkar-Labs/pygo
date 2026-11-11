package tensor

import "fmt"

func Fused0130ReluAddCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_add_cos: nil operand")
	}
	r := x.Add(y)
	r = r.ReLU()
	r = r.Cos()
	return r, nil
}
func Fused0130ReluAddCosName() string { return "relu_add_cos" }
