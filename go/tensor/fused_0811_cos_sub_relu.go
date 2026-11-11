package tensor

import "fmt"

func Fused0811CosSubRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_sub_relu: nil operand")
	}
	r := x.Sub(y)
	r = r.Cos()
	r = r.ReLU()
	return r, nil
}
func Fused0811CosSubReluName() string { return "cos_sub_relu" }
