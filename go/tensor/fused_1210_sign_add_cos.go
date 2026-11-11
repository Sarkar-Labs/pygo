package tensor

import "fmt"

func Fused1210SignAddCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_add_cos: nil operand")
	}
	r := x.Add(y)
	r = r.Sign()
	r = r.Cos()
	return r, nil
}
func Fused1210SignAddCosName() string { return "sign_add_cos" }
