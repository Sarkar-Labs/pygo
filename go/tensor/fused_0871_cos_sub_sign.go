package tensor

import "fmt"

func Fused0871CosSubSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_sub_sign: nil operand")
	}
	r := x.Sub(y)
	r = r.Cos()
	r = r.Sign()
	return r, nil
}
func Fused0871CosSubSignName() string { return "cos_sub_sign" }
