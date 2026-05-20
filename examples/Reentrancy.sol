// SPDX-License-Identifier: MIT
pragma solidity ^0.5.12;

contract Reentrance {
    mapping(address => uint256) public balances;

    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }

    function withdraw(uint256 amount) public {
        require(balances[msg.sender] >= amount, "insufficient");
        (bool ok, ) = msg.sender.call.value(amount)("");
        require(ok, "transfer failed");
        balances[msg.sender] -= amount;
    }
}
