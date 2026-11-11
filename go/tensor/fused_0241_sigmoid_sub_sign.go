package tensor

import "fmt"

func Fused0241SigmoidSubSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_sub_sign: nil operand")
	}
	r := x.Sub(y)
	r = r.Sigmoid()
	r = r.Sign()
	return r, nil
}
func Fused0241SigmoidSubSignName() string { return "sigmoid_sub_sign" }
