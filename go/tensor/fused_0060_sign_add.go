package tensor

import "fmt"

func Fused0060SignAdd(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_add: nil operand")
	}
	r := x.Add(y)
	r = r.Sign()
	return r, nil
}
func Fused0060SignAddName() string { return "sign_add" }
