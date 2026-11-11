package tensor

import "fmt"

func Fused0131ReluSubCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_sub_cos: nil operand")
	}
	r := x.Sub(y)
	r = r.ReLU()
	r = r.Cos()
	return r, nil
}
func Fused0131ReluSubCosName() string { return "relu_sub_cos" }
