package tensor

import "fmt"

func Fused1211SignSubCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_sub_cos: nil operand")
	}
	r := x.Sub(y)
	r = r.Sign()
	r = r.Cos()
	return r, nil
}
func Fused1211SignSubCosName() string { return "sign_sub_cos" }
