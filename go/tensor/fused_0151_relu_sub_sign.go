package tensor

import "fmt"

func Fused0151ReluSubSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_sub_sign: nil operand")
	}
	r := x.Sub(y)
	r = r.ReLU()
	r = r.Sign()
	return r, nil
}
func Fused0151ReluSubSignName() string { return "relu_sub_sign" }
