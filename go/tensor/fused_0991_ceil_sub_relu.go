package tensor

import "fmt"

func Fused0991CeilSubRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_sub_relu: nil operand")
	}
	r := x.Sub(y)
	r = r.Ceil()
	r = r.ReLU()
	return r, nil
}
func Fused0991CeilSubReluName() string { return "ceil_sub_relu" }
