package tensor

import "fmt"

func Fused1250SignAddLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_add_log2: nil operand")
	}
	r := x.Add(y)
	r = r.Sign()
	r = r.Log2()
	return r, nil
}
func Fused1250SignAddLog2Name() string { return "sign_add_log2" }
